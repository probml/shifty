
# skax is sklearn in jax

# numiter  * batchsize = numepochs * ntraindef 


import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import einops

from functools import partial
from collections import namedtuple
import jax
import jax.random as jr
import jax.numpy as jnp
from jax import vmap, grad, jit
#import jax.debug
import itertools
from itertools import repeat
from time import time
import chex
import typing

import jax
from typing import Any, Callable, Sequence
from jax import lax, random, numpy as jnp
from flax.core import freeze, unfreeze
from flax import linen as nn
import flax
from flax.training import train_state

import jaxopt
import optax
import distrax
from jaxopt import OptaxSolver
import tensorflow as tf
import tensorflow_datasets as tfds

from sklearn.base import ClassifierMixin

def loglikelihood_fn(params, model, X, y):
    # 1/N sum_n log p(yn | xn, params)
    logits = model.apply(params, X)
    return jnp.mean(distrax.Categorical(logits).log_prob(y))

def logprior_fn(params, sigma):
    # log p(params)
    leaves, _ = jax.tree_util.tree_flatten(params)
    flat_params = jnp.concatenate([jnp.ravel(a) for a in leaves])
    return jnp.sum(distrax.Normal(0, sigma).log_prob(flat_params))

@partial(jax.jit, static_argnames=["network"])
def objective(params, data, network, prior_sigma, ntrain): 
    # objective = -1/N [ (sum_n log p(yn|xn, theta)) + log p(theta) ]
    X, y = data["X"], data["y"]
    logjoint = loglikelihood_fn(params, network, X, y) + (1/ntrain)*logprior_fn(params, prior_sigma)
    return -logjoint


def make_batched_dataset(X, y, batch_size):
    # Convert dataset into a fixed set of minibatches 
    # usage: for batch in ds
    ds = tf.data.Dataset.from_tensor_slices({"X": X, "y": y})
    ds = ds.batch(batch_size)
    ds = tfds.as_numpy(ds)
    return ds

def make_data_iterator(X, y, batch_size):
    # Convert dataset into an infinite stream of minibatches 
    # usage: for i in nsteps: batch = next(iterator)
    ntrain = X.shape[0]
    ds = tf.data.Dataset.from_tensor_slices({"X": X, "y": y})
    # https://jaxopt.github.io/stable/auto_examples/deep_learning/haiku_image_classif.htm
    ds = ds.cache().repeat()
    #ds = ds.shuffle(10 * batch_size, seed=0) # how use key?
    ds = ds.shuffle(ntrain)
    ds = ds.batch(batch_size)
    iterator = iter(tfds.as_numpy(ds)) #ds.as_numpy_iterator()
    return iterator

class LogRegNetwork(nn.Module):
    nclasses: int

    @nn.compact
    def __call__(self, x):
        logits = nn.Dense(self.nclasses)(x)
        return logits

class MLPNetwork(nn.Module):
  nfeatures_per_layer: Sequence[int]

  @nn.compact
  def __call__(self, inputs):
    x = inputs
    nlayers = len(self.nfeatures_per_layer)
    for i, feat in enumerate(self.nfeatures_per_layer):
      x = nn.Dense(feat, name=f'layers_{i}')(x)
      if i != (nlayers - 1):
        x = nn.relu(x)
    return x


class NeuralNetClassifier(ClassifierMixin):
    def __init__(self, network, key, nclasses, *,  l2reg=1e-5,
                optimizer = 'lbfgs', batch_size=128, max_iter=1000, num_epochs=5):
        # optimizer is {'lbfgs', 'polyak', 'armijo'} or an optax object
        self.nclasses = nclasses
        self.network = network
        self.optimization_results = None
        self.max_iter = max_iter
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.l2reg = l2reg
        #x = jr.normal(key, (ninputs,)) # single random input 
        #self.params = self.network.init(key, x)
        self.params = None # must first call fit
        self.key = key

    def predict(self, inputs):
        return jax.nn.softmax(self.network.apply(self.params, inputs))

    def fit(self, X, y):
        ninputs = X.shape[1]
        x = jr.normal(self.key, (ninputs,)) # single random input 
        self.params = self.network.init(self.key, x)
        if isinstance(self.optimizer, str) and (self.optimizer.lower() == "lbfgs"):
            return self.fit_bfgs(self.key, X, y)
        else:
            return self.fit_jaxopt(self.key, X, y)

    def fit_bfgs(self, key, X, y):
        # Full batch optimization
        sigma = np.sqrt(1/self.l2reg)
        N = X.shape[0]
        data = {"X": X, "y": y}
        def loss_fn(params):
            return objective(params=params, data=data,  network=self.network,  prior_sigma=sigma, ntrain=N)
        solver = jaxopt.LBFGS(fun=loss_fn, maxiter=self.max_iter)
        res = solver.run(self.params)
        self.params = res.params
        self.optimization_results = res


    def fit_jaxopt(self, key, X, y):
        # https://jaxopt.github.io/stable/auto_examples/deep_learning/flax_resnet.html
        # https://github.com/blackjax-devs/blackjax/discussions/360#discussioncomment-3756412
        sigma = np.sqrt(1/self.l2reg)
        ntrain, B = X.shape[0], self.batch_size

        @jax.jit
        def loss_fn(params, data):
            # objective = -1/N [ (sum_n log p(yn|xn, theta)) + log p(theta) ]
            X, y = data["X"], data["y"]
            logits = self.network.apply(params, X)
            loglik = jnp.mean(distrax.Categorical(logits).log_prob(y))
            logjoint = loglik + (1/ntrain)*logprior_fn(params, sigma)
            return -logjoint

        if isinstance(self.optimizer, str) and (self.optimizer.lower() == "polyak"):
            solver = jaxopt.PolyakSGD(fun=loss_fn, maxiter=self.max_iter)
        elif isinstance(self.optimizer, str) and (self.optimizer.lower() == "armijo"):
            solver = jaxopt.ArmijoSGD(fun=loss_fn, maxiter=self.max_iter)
        else:
            solver = OptaxSolver(opt=self.optimizer, fun=loss_fn, maxiter=self.max_iter)
    
        iterator = make_data_iterator(X, y, self.batch_size)
        res = solver.run_iterator(self.params, iterator=iterator)
        self.params = res.params

    def fit_optax(self, key, X, y):
        # Based on https://github.com/google/flax/blob/main/examples/mnist/train.py
        # https://github.com/google/flax/blob/main/examples/vae/train.py
        sigma = np.sqrt(1/self.l2reg)
        ntrain = X.shape[0]
        ds = make_data_iterator(X, y, self.batch_size)

        state = train_state.TrainState.create(
            apply_fn=self.network.apply, params=self.params['params'], tx=self.optimizer)

        @jax.jit
        def train_step(state, data):
            X, y = data["X"], data["y"]
            # Computes gradients, loss and accuracy for a single batch.
            # loss = -1/N [ (sum_n log p(yn|xn, theta)) + log p(theta) ]
            def loss_fn(params):
                logits = state.apply_fn({'params': params}, X)
                loglik = jnp.mean(distrax.Categorical(logits).log_prob(y))
                logjoint = loglik + (1/ntrain)*logprior_fn(params, sigma)
                loss = -logjoint
                return loss
            loss, grads = jax.grad(loss_fn, has_aux=True)(state.params)
            return state.apply_gradients(grads=grads), loss

        # main loop
        steps_per_epoch = ntrain / self.batch_size
        for epoch in range(self.num_epochs):
            epoch_loss = []
            for _ in range(steps_per_epoch):
                batch = next(ds)
                #rng, key = random.split(rng)
                state, loss = train_step(state, batch)  
                epoch_loss.append(loss)
            train_loss = np.mean(epoch_loss)
            print('epoch {:d}, train loss {:0.3f}'.format(epoch, train_loss))

        self.params = {'params': state.params}

        
      
        
