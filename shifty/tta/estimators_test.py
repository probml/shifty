import pytest

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import einops
import matplotlib
from functools import partial
from collections import namedtuple
import dataclasses


import jax
import jax.random as jr
import jax.scipy as jsp 
import jax.numpy as jnp
from jax import vmap, grad, jit
import chex
import typing
from copy import deepcopy
from collections import Counter

from shifty.tta.data_generator import *
from shifty.tta.data_generator_test import *
from shifty.tta.label_space import *
from shifty.tta.estimators import *
from shifty.skax.skax import *


def test_oracle():
    print('test_oracle')
    key = jr.PRNGKey(0)
    corr_source = 0.3
    corr_targets = jnp.arange(0.1, 0.9, step=0.1)
    ls = LabelSpace(nclasses=2, nfactors=2)
    #src_dist = GMMDataGenerator(key, corr_source, ls, nfeatures=4)
    src_dist = NurdDataGenerator(key, corr_source, ls, sf=5)

    est = OracleEstimator(ls)
    metrics =  evaluate_estimator(key, src_dist, corr_targets, est)
    print(metrics)
    assert jnp.allclose(metrics, jnp.zeros(len(corr_targets)))


def plot_posteriors_old(post_dict, nmax=100):
  post1 = list(post_dict.values())[0]
  ntrain, nmix = post1.shape
  fig, axs = plt.subplots(1, len(post_dict), figsize=(10,5))
  # select a subset of points to plot
  n = np.min([ntrain, nmax])
  ndx = np.round(np.linspace(1, ntrain-1, num=n)).astype(int)
  m = 0
  for name, post in post_dict.items():
    ax = axs[m]
    m += 1
    ax.plot(post[ndx])
    ax.set_title(name)


def plot_posteriors(post_dict, nmax=100):
  post1 = list(post_dict.values())[0]
  ntrain, nmix = post1.shape
  #fig, axs = plt.subplots(1, nmix, figsize=(10,5))
  fig, axs = plt.subplots(2, 2, figsize=(10,10))
  #axs = axs.reshape(-1)
  axs = axs.flatten()
  n = np.min([ntrain, nmax])
  ndx = np.round(np.linspace(1, ntrain-1, num=n)).astype(int)   # select a subset of points to plot
  for m in range(nmix):
    ax = axs[m]
    for name, post in post_dict.items():
      ax.plot(post[ndx,m], label=name)
    ax.legend()


def eval_model_nurd_grid(model, name):
    key = jr.PRNGKey(42)
    corr_source = 0.3
    ls = LabelSpace(nclasses=2, nfactors=2)
    #src_dist = GMMDataGenerator(key, corr_source, ls, nfeatures=4)
    src_dist = NurdDataGenerator(key, corr_source, ls, sf=1)

    X_train, y, a = src_dist.sample(key, 500)
    mix_train = ls.flatten_labels(y, a)
    mix_post_true = src_dist.predict_joint(X_train)

    model.fit(X_train, mix_train)

    # evaluate on a dense grid
    xs, x1, x2 = make_xgrid(npoints = 100)
    mix_post_true = src_dist.predict_joint(xs)
    mix_post_pred = model.predict(xs)

    print('max error of {} = {:.3f}'.format(name, jnp.max(mix_post_true - mix_post_pred)))
    plot_posteriors({'true': mix_post_true, name: mix_post_pred})

def eval_models_nurd_grid():
    # see how well various classifiers can predict the mixture source for the nurd data
    key = jr.PRNGKey(42)
    ntargets = 4
    nhidden =  (10, 10, ntargets)
    network = MLPNetwork(nhidden)
    model = NeuralNetClassifier(network, key, ntargets, l2reg=1e-2, optimizer = "adam+warmup", 
            batch_size=32, num_epochs=20, print_every=0) 
    eval_model_nurd_grid(model, 'mlp')

    nhidden =  (ntargets,) 
    network = MLPNetwork(nhidden)
    model = NeuralNetClassifier(network, key, ntargets, l2reg=1e-5, optimizer = "adam+warmup", 
            batch_size=32, num_epochs=20, print_every=0) 
    eval_model_nurd_grid(model, 'logreg')

    nhidden =  (ntargets,) 
    network = MLPNetwork(nhidden)
    model = NeuralNetClassifier(network, key, ntargets, l2reg=1e-5, optimizer = "lbfgs", 
            batch_size=32, num_epochs=20, print_every=0) 
    eval_model_nurd_grid(model, 'logreg+lbfgs')

def eval_em(key, src_dist, target_dist, ls, clf, name, nsamples_src=500, nsamples_target=100):
    prior_source = src_dist.params.prior
    prior_target = target_dist.params.prior
    key, subkey = jr.split(key)
    Xs, Ys, As = src_dist.sample(subkey, nsamples_src)
    key, subkey = jr.split(key)
    Xt, Yt, At = target_dist.sample(subkey, nsamples_target)
    est = EMEstimator(clf, ls)
    est.fit_source(Xs, Ys, As, src_dist)
    est.fit_target(Xt, target_dist)
    src_delta = jnp.max(prior_source - est.prior_source)
    target_delta = jnp.max(prior_target - est.prior_target)
    print('eval em ', name)
    print('true prior source:', prior_source, 'est:', est.prior_source, 'delta:', src_delta, 'correlation:', src_dist.correlation)
    print('true prior target:', prior_target, 'est:', est.prior_target, 'delta:', target_delta,  'correlation:', target_dist.correlation)
    assert src_delta < 0.2
    assert target_delta < 0.2

def eval_em_nurd():
    key = jr.PRNGKey(42)
    corr_source = 0.3
    ls = LabelSpace(nclasses=2, nfactors=2)
    src_dist = NurdDataGenerator(key, corr_source, ls, sf=1)

    rhos = [0.1, 0.5, 0.9]
    for rho in rhos:
        target_dist = deepcopy(src_dist)
        target_dist.shift_prior_correlation(rho)
        clf = make_logreg()
        eval_em(key, src_dist, target_dist, ls, clf, 'logreg')
        clf = make_mlp()
        eval_em(key, src_dist, target_dist, ls, clf, 'mlp')


    

