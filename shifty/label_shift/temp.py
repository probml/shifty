
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import einops
import matplotlib
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

import jaxopt
import optax
import distrax
from jaxopt import OptaxSolver
import tensorflow as tf

import sklearn.datasets
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import LogisticRegression

logistic_loss = jax.vmap(jaxopt.loss.multiclass_logistic_loss)

def regularizer(params, l2reg):
    sqnorm = jaxopt.tree_util.tree_l2_norm(params, squared=True)
    return 0.5 * l2reg * sqnorm

def loss_from_logits(params, l2reg, logits, labels):
    mean_loss = jnp.mean(logistic_loss(labels, logits))
    return mean_loss + regularizer(params, l2reg)

def loglikelihood_fn(params, model, X, y):
    logits = model.apply(params, X)
    return jnp.mean(distrax.Categorical(logits).log_prob(y))

def logprior_fn(params, sigma):
    leaves, _ = jax.tree_util.tree_flatten(params)
    flat_params = jnp.concatenate([jnp.ravel(a) for a in leaves])
    return jnp.sum(distrax.Normal(0, sigma).log_prob(flat_params))


class LogRegModule(nn.Module):
  nclasses: int
  W_init_fn: Any
  b_init_fn: Any

  @nn.compact
  def __call__(self, x):
    if self.W_init_fn is not None:
      logits = nn.Dense(self.nclasses, kernel_init=self.W_init_fn, bias_init=self.b_init_fn)(x)
    else:
      logits = nn.Dense(self.nclasses)(x)
    return logits

class LogReg:
  def __init__(self, key, ninputs, nclasses, *,  max_iter = 100, l2reg=1e-5,
                optimizer='lbfgs', batch_size = 32, W_init=None, b_init=None):
    self.nclasses = nclasses
    if W_init is not None: # specify initial parameters by hand
      W_init_fn = lambda key, shape, dtype: W_init
      b_init_fn = lambda key, shape, dtype: b_init
      self.model = LogRegModule(nclasses, W_init_fn, b_init_fn)
    else:
      self.model = LogRegModule(nclasses, None, None)
    self.optimization_results = None
    self.max_iter = max_iter
    self.l2reg = l2reg
    self.optimizer = optimizer
    self.batch_size = batch_size
    x = jr.normal(key, (ninputs,)) # single random input 
    self.params = self.model.init(key, x)
    
  def predict(self, inputs):
    return jax.nn.softmax(self.model.apply(self.params, inputs))

    def fit_batch(self, X, y):
        sigma = np.sqrt(1/self.l2reg)
        @jax.jit
        def objective(params, data): 
            X, y = data["X"], data["y"]
            logjoint = loglikelihood_fn(params, self.model, X, y) + logprior_fn(params, sigma)
            return -logjoint

        data = {"X": X, "y": y}
        solver = jaxopt.LBFGS(fun=partial(objective, data=data), maxiter=self.max_iter)
        res = solver.run(self.params)
        self.params = res.params
        self.optimization_results = res


    def fit2(self, X, y):
        # https://jaxopt.github.io/stable/auto_examples/deep_learning/flax_resnet.html
        # https://github.com/blackjax-devs/blackjax/discussions/360#discussioncomment-3756412
        sigma = np.sqrt(1/self.l2reg)
        @jax.jit
        def objective(params, data): 
            X, y = data["X"], data["y"]
            logjoint = loglikelihood_fn(params, self.model, X, y) + logprior_fn(params, sigma)
            return -logjoint

        # Convert dataset into a stream of minibatches (for stochasitc optimizers)
        # https://www.tensorflow.org/api_docs/python/tf/data/Dataset?version=nightly#from_tensor_slices
        ds = tf.data.Dataset.from_tensor_slices({"X": X, "y": y})
        # https://jaxopt.github.io/stable/auto_examples/deep_learning/haiku_image_classif.htm
        ds = ds.cache().repeat()
        ds = ds.shuffle(10 * self.batch_size, seed=0)
        ds = ds.batch(self.batch_size)
        iterator = ds.as_numpy_iterator()

        if self.optimizer.lower() == 'lbfgs':
            data = {"X": X, "y": y}
            solver = jaxopt.LBFGS(fun=partial(objective, data=data), maxiter=self.max_iter)
            res = solver.run(self.params)
        elif self.optimizer.lower() == "polyak":
            solver = jaxopt.PolyakSGD(maxiter=self.max_iter)
        else:
            solver = OptaxSolver(opt=self.optimizer, fun=objective, maxiter=self.max_iter)
    
            res = solver.run_iterator(self.params, iterator=iterator)
        self.params = res.params
 



