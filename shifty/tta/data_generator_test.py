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
from shifty.tta.label_space import *

def test_gmm():
    print('test_gmm')
    key = jr.PRNGKey(0)
    class_prior = jnp.array([0.3, 0.7])
    ls = LabelSpace(nclasses=2, nfactors=2)
    gmm_source = GMMDataGenerator(key, correlation=0.1, label_space=ls, nfeatures=10, class_prior=class_prior)
    gmm_target = deepcopy(gmm_source)
    gmm_target.shift_prior_correlation(0.9)
    prior_y, prior_a = gmm_source.label_space.unflatten_dist(gmm_source.params.prior)
    assert np.allclose(prior_y, class_prior)
    prior_y, prior_a = gmm_target.label_space.unflatten_dist(gmm_target.params.prior)
    assert np.allclose(prior_y, class_prior)


def test_gmm2():
    print('test_gmm2')
    key = jr.PRNGKey(0)
    class_prior = jnp.array([0.3, 0.7])
    ls = LabelSpace(nclasses=2, nfactors=2)
    gmm = GMMDataGenerator(key, correlation=0.1,  label_space=ls, nfeatures=10, class_prior=class_prior)
    prior_y, prior_a = gmm.label_space.unflatten_dist(gmm.params.prior)
    assert np.allclose(prior_y, class_prior)

    n = 1000
    X, y, a = gmm.sample(key, n)
    counts = jnp.sum(y==1); prior_y_mle = counts/n;
    counts = jnp.sum(a==1); prior_a_mle = counts/n;

    assert np.allclose(prior_y_mle, prior_y[1], atol=1e-1)
    assert np.allclose(prior_a_mle, prior_a[1], atol=1e-1)


def yz_to_mix(y, z):
    nclasses, nfactors = 2, 2
    m = jnp.ravel_multi_index((jnp.array([y]), jnp.array([z])), (nclasses, nfactors))
    return m[0]

def mix_to_yz(m):
    nclasses, nfactors = 2, 2
    yz = jnp.unravel_index(m, (nclasses, nfactors))
    return yz[0], yz[1]

def make_xgrid(npoints = 100):
    npoints = npoints * 1j
    xyrange = jnp.array([[-3, 3], [-3, 3]])
    mesh = jnp.mgrid[xyrange[0, 0] : xyrange[0, 1] : npoints, xyrange[1, 0] : xyrange[1, 1] : npoints]
    x1, x2 = mesh[0], mesh[1]
    points = jnp.vstack([jnp.ravel(x1), jnp.ravel(x2)]).T
    return points, x1, x2


def _plot_class_cond_dist(dist):
    assert dist.nmix == 4
    assert dist.nfeatures == 2
    fig, axs = plt.subplots(2, 2, figsize=(10,10))
    points, x1, x2 = make_xgrid(100)
    for z, ax in enumerate(axs.flat):
        p = dist.lik_fn(z, points).reshape(x1.shape[0], x2.shape[0])
        contour = ax.contourf(x1, x2, p)
        # cbar = fig.colorbar(contour, ax=ax)
        y, a = dist.label_space.unflatten_labels(z)
        ax.set_title('y={:d}, a={:d}'.format(y, a))

def plot_nurd_distributions():
    key = jr.PRNGKey(0)
    corr_source = 0.2
    ls = LabelSpace(nclasses=2, nfactors=2)
    src_dist = NurdDataGenerator(key, corr_source, ls, b=1, sf=1)
    _plot_class_cond_dist(src_dist)


def _plot_mix_post(mix_post, nmax=100, ttl=None):
  ntrain, nmix = mix_post.shape
  # select a subset of points to plot
  n = np.min([ntrain, nmax])
  ndx = np.round(np.linspace(1, ntrain-1, num=n)).astype(int)
  colors = ['r', 'g', 'b', 'k']
  plt.figure()
  for m in range(nmix):
    y, z = mix_to_yz(m)
    plt.plot(mix_post[ndx,m], colors[m], label='m={:d},y={:d},z={:d}'.format(m,y,z))
  plt.legend()
  if ttl is not None:
    plt.title(ttl)

def test_nurd_post():
    print('test_nurd_post')
    key = jr.PRNGKey(0)
    corr_source = 0.1
    ls = LabelSpace(nclasses=2, nfactors=2)
    src_dist = NurdDataGenerator(key, corr_source, ls, b=1, sf=1)
    xs, x1, x2 = make_xgrid(npoints = 100)
    mix_post = src_dist.predict_joint(xs)
    class_post = src_dist.predict_class(xs)
    print(mix_post)
    _plot_mix_post(mix_post)


def normalize_vec(v):
    return v / jnp.sum(v)

def counter_to_freqs(counter, n):
    c = np.zeros(n)
    for i in range(n):
        c[i]= counter[i]
    return normalize_vec(c)

def test_nurd_sampler():
    print('test_nurd_sampler')
    key = jr.PRNGKey(0)
    corr_source = 0.3
    ls = LabelSpace(nclasses=2, nfactors=2)
    src_dist = NurdDataGenerator(key, corr_source, ls, b=1, sf=1)

    X, y, a = src_dist.sample(key, 5000)
    mix_train = ls.flatten_labels(y, a)

    mix_post = src_dist.predict_joint(X)
    mix_pred = np.argmax(mix_post, axis=1) # MAP estimate of predicted label

    counter_true = Counter(np.array(mix_train))
    counts_true = counter_to_freqs(counter_true, 4)
    print(counts_true)

    counter_pred = Counter(np.array(mix_pred))
    counts_pred = counter_to_freqs(counter_pred, 4)
    print(counts_pred)

    assert np.allclose(counts_true, counts_pred, atol=1e-1)

    # compute mixture posterior on some of the sampled data
    # then derive the MAP estimate for the mix labels
    # this should be similar to the sampled mix labels

    ntrain = X.shape[0]
    n = np.min([ntrain, 10])
    ndx = np.round(np.linspace(1, ntrain-1, num=n)).astype(int)
    mix_true =  mix_train[ndx]

    mix_post = src_dist.predict_joint(X[ndx])
    mix_hat = np.argmax(mix_post, axis=1)

    print(mix_true)
    print(mix_hat)
    print(mix_post)