# to show output from the 'tests', run with 
# pytest skax_test.py  -rP

from functools import partial
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=3)
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

import sklearn.datasets
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import LogisticRegression

from skax import *
#jax.config.update("jax_enable_x64", True) # jaxopt.lbfgs uses float32

def make_test_data():
    iris = sklearn.datasets.load_iris()
    X = iris["data"]
    #y = (iris["target"] == 2).astype(np.int)  # 1 if Iris-Virginica, else 0'
    y = iris["target"]
    nclasses = len(np.unique(y)) # 
    ndata, ndim = X.shape  # 150, 4
    key = jr.PRNGKey(0)
    noise = jr.normal(key, (ndata, ndim)) * 2
    X = X + noise # add noise to make the classes less separable
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    return X, y

def compute_mle(X, y):
    # We set C to a large number to turn off regularization.
    # We don't fit the bias term to simplify the comparison below.
    log_reg = LogisticRegression(solver="lbfgs", C=1e5, fit_intercept=True)
    log_reg.fit(X, y)
    W_mle = log_reg.coef_ # (nclasses, ndim)
    b_mle = log_reg.intercept_ # (nclasses,)
    true_probs = log_reg.predict_proba(X)
    return true_probs, W_mle, b_mle


def compare_method(optimizer, name=None, batch_size=None, max_iter=500):
    X, y = make_test_data()
    true_probs, W_mle, b_mle = compute_mle(X, y)
    nclasses, ndim = W_mle.shape
    key = jr.PRNGKey(0)
    l2reg = 1e-5
    #network = LogRegNetwork(nclasses = nclasses)
    #network = MLPNetwork((5, nclasses,))
    network = MLPNetwork((nclasses,)) # no hidden layers
    model = NeuralNetClassifier(network, key, nclasses, l2reg=l2reg, optimizer = optimizer, batch_size=batch_size, max_iter=max_iter) 
    #model = LogReg(key, nclasses, max_iter=max_iter, l2reg=l2reg, optimizer=optimizer, batch_size=batch_size)  
    model.fit(X, y)
    probs = np.array(model.predict(X))
    print('method {:s}, max deviation from true probs {:.3f}'.format(name, np.max(true_probs - probs)))
    print('truth: ', true_probs[0])
    print('pred: ', probs[0])


def test_bfgs():
    compare_method("lbfgs", name="lbfgs")

def test_adam_full_batch():
    X, y = make_test_data()
    ntrain = X.shape[0]
    compare_method(optax.adam(0.01), name="adam 0.01, bs=N", batch_size=ntrain)

def test_adam():
    compare_method(optax.adam(0.01), name="adam 0.01, bs=32", batch_size=32)

def test_polyak():
    compare_method("polyak", name="polyak, bs=32", batch_size=32)

def test_armijo():
    compare_method("armijo", name="polyak, bs=32", batch_size=32)
