
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

def logprior_fn(params, sigma):
    # log p(params)
    leaves, _ = jax.tree_util.tree_flatten(params)
    flat_params = jnp.concatenate([jnp.ravel(a) for a in leaves])
    return jnp.sum(distrax.Normal(0, sigma).log_prob(flat_params))

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
    ds = ds.shuffle(10 * batch_size, seed=0) # how use key?
    #ds = ds.shuffle(ntrain)
    ds = ds.batch(batch_size)
    iterator = iter(tfds.as_numpy(ds)) #ds.as_numpy_iterator()
    return iterator

@partial(jax.jit, static_argnums=(1,2))
def get_batch_train_ixs(key, num_train, batch_size):
    # return indices of training set in a random order
    steps_per_epoch = num_train // batch_size
    batch_ixs = jax.random.permutation(key, num_train)
    batch_ixs = batch_ixs[:steps_per_epoch * batch_size]
    batch_ixs = batch_ixs.reshape(steps_per_epoch, batch_size)
    return batch_ixs


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
    def __init__(self, network, key, nclasses, *,  l2reg=1e-5, standardize = True,
                optimizer = 'lbfgs', batch_size=128, max_iter=1000, num_epochs=10, print_every=0):
        # optimizer is {'lbfgs', 'polyak', 'armijo', 'adam+warmup'} or an optax object
        self.nclasses = nclasses
        self.network = network
        self.standardize = standardize
        self.max_iter = max_iter
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.l2reg = l2reg
        self.print_every = print_every
        self.params = None # must first call fit
        self.key = key

    def predict(self, X):
        if self.standardize:
            X = X - self.mean
            X = X / self.std
        return jax.nn.softmax(self.network.apply(self.params, X))

    def fit(self, X, y):
        if self.params is None:
            nfeatures = X.shape[1]
            x = jr.normal(self.key, (nfeatures,)) # single random input 
            self.params = self.network.init(self.key, x)
        if self.standardize:
            self.mean = jnp.mean(X, axis=0)
            self.std = jnp.std(X, axis=0) + 1e-5 
            X = X - self.mean
            X = X / self.std
        ntrain = X.shape[0]
        if isinstance(self.optimizer, str) and (self.optimizer.lower() == "lbfgs"):
            return self.fit_bfgs(self.key, X, y)
        elif isinstance(self.optimizer, str) and (self.optimizer.lower() == "polyak"):
            return self.fit_jaxopt(self.key, X, y)
        elif isinstance(self.optimizer, str) and (self.optimizer.lower() == "armijo"):
            return self.fit_jaxopt(self.key, X, y)
        elif isinstance(self.optimizer, str) and (self.optimizer.lower() == "adam+warmup"):
            total_steps = self.num_epochs*(ntrain//self.batch_size)  
            warmup_cosine_decay_scheduler = optax.warmup_cosine_decay_schedule(
                init_value=1e-3, peak_value=1e-1, warmup_steps=int(total_steps*0.1),
                decay_steps=total_steps, end_value=1e-3)
            self.optimizer = optax.adam(learning_rate=warmup_cosine_decay_scheduler)
            return self.fit_optax(self.key, X, y)
        else:
            #return self.fit_jaxopt(self.key, X, y)
            return self.fit_optax(self.key, X, y)

    def fit_bfgs(self, key, X, y):
        # Full batch optimization
        sigma = np.sqrt(1/self.l2reg)
        ntrain = X.shape[0]
        @jax.jit
        def loss_fn(params):
            # objective = -1/N [ (sum_n log p(yn|xn, theta)) + log p(theta) ]
            logits = self.network.apply(params, X)
            loglik = jnp.mean(distrax.Categorical(logits).log_prob(y))
            logjoint = loglik + (1/ntrain)*logprior_fn(params, sigma)
            return -logjoint
        solver = jaxopt.LBFGS(fun=loss_fn, maxiter=self.max_iter)
        res = solver.run(self.params)
        self.params = res.params

    def fit_jaxopt(self, key, X, y):
        # https://jaxopt.github.io/stable/auto_examples/deep_learning/flax_resnet.html
        sigma = np.sqrt(1/self.l2reg)
        ntrain = X.shape[0]

        @jax.jit
        def loss_fn(params, data):
            # objective = -1/N [ (sum_n log p(yn|xn, theta)) + log p(theta) ]
            # https://github.com/blackjax-devs/blackjax/discussions/360#discussioncomment-3756412
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
 
        @jax.jit
        def train_step(key, state, Xb, yb):
            # Computes gradients, loss and accuracy for a single batch.
            # loss = -1/N [ (sum_n log p(yn|xn, theta)) + log p(theta) ]
            def loss_fn(params):
                logits = state.apply_fn({'params': params}, Xb)
                loglik = jnp.mean(distrax.Categorical(logits).log_prob(yb))
                logjoint = loglik + (1/ntrain)*logprior_fn(params, sigma)
                return -logjoint
            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            return loss, state.apply_gradients(grads=grads)

        def train_epoch(key, state):
            key, sub_key = jr.split(key)
            batch_ixs = get_batch_train_ixs(sub_key, ntrain, self.batch_size)
            num_batches = len(batch_ixs)
            key, sub_key = jr.split(key)
            keys = jax.random.split(sub_key, num_batches)    
            total_loss = 0
            for key, batch_ix in zip(keys, batch_ixs):
                X_batch, y_batch = X[batch_ix], y[batch_ix]
                loss, state = train_step(key, state, X_batch, y_batch)
                total_loss += loss
            return total_loss.item(), state


        # main loop
        state = train_state.TrainState.create(
            apply_fn=self.network.apply, params=self.params['params'], tx=self.optimizer)
        for epoch in range(self.num_epochs):
            key, sub_key = jr.split(key)
            train_loss, state = train_epoch(sub_key, state)
            if (self.print_every > 0) and (epoch % self.print_every == 0):
                print('epoch {:d}, train loss {:0.3f}'.format(epoch, train_loss))

        self.params = {'params': state.params}

        
      
        
