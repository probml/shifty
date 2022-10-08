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

def print_probs(probs):
    str = ['{:0.3f}'.format(p) for p in probs]
    print(str)

def make_random_data(seed, n_samples, class_sep=0.5, n_features=20):
    X, y = sklearn.datasets.make_classification(n_samples=500, n_features=n_features,  n_informative=10,
    n_redundant=5, n_repeated=0,
    n_classes=10, n_clusters_per_class=1, weights=None, flip_y=0.01, class_sep=class_sep,
    hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=seed)
    return X, y



def compute_mle(X, y):
    # Use sklearn to fit logreg model
    l2reg = 1e-5
    classifier = Pipeline([
            ('standardscaler', StandardScaler()), 
            ('logreg', LogisticRegression(random_state=0, max_iter=100, C=1/l2reg))])
    classifier.fit(X, y)
    true_probs = classifier.predict_proba(X)
    return true_probs


def compare_method(optimizer, name=None, batch_size=None, num_epochs=50,
                    data_seed=0, n_samples=500, class_sep=0.5):
    key = jr.PRNGKey(0)
    X, y = make_random_data(data_seed, n_samples, class_sep)
    true_probs = compute_mle(X, y)
    nclasses = len(np.unique(y))
    if batch_size is None:
        batch_size = n_samples # full batch
    l2reg = 1e-5
    #network = MLPNetwork((5, nclasses,))
    network = MLPNetwork((nclasses,)) # no hidden layers == logistic regression
    model = NeuralNetClassifier(network, key, nclasses, l2reg=l2reg, optimizer = optimizer, 
            batch_size=batch_size, num_epochs=num_epochs, print_every=0)  
    model.fit(X, y)
    probs = np.array(model.predict(X))
    print('method {:s}, max deviation from true probs {:.3f}'.format(name, np.max(true_probs - probs)))
    print('truth'); print_probs(true_probs[0])
    print('pred'); print_probs(probs[0])


def s_test_bfgs():
    compare_method("lbfgs", name="lbfgs")

def s_test_armijo():
    compare_method("armijo", name="armijo, bs=32", batch_size=32)

def s_test_adam_const():
    compare_method(optax.adam(1e-3), name="adam 1e-3, bs=32", batch_size=32)

def test_adam_warmup():
    compare_method("adam+warmup", name="adam+warmup, bs=32", batch_size=32)

def test_adam_warmup1():
    data_seed=1; n_samples=500; class_sep=0.5
    compare_method("adam+warmup", name="adam+warmup, bs=32", batch_size=32,
        data_seed=data_seed, n_samples=n_samples, class_sep=class_sep)

def test_adam_warmup2():
    data_seed=1; n_samples=500; class_sep=1
    compare_method("adam+warmup", name="adam+warmup, bs=32", batch_size=32,
        data_seed=data_seed, n_samples=n_samples, class_sep=class_sep)

def test_adam_warmup2():
    data_seed=1; n_samples=100; class_sep=2
    compare_method("adam+warmup", name="adam+warmup, bs=32", batch_size=32,
        data_seed=data_seed, n_samples=n_samples, class_sep=class_sep)

def test_adam_warmup3():
    data_seed=1; n_samples=500; class_sep=0.5; n_features = 100
    compare_method("adam+warmup", name="adam+warmup, bs=32", batch_size=32,
        data_seed=data_seed, n_samples=n_samples, class_sep=class_sep, n_features=n_features)