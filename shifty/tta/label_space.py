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

def predict_bayes(prior, likelihood_fn, X):
    nmix = len(prior)
    lik_fn = partial(likelihood_fn, X=X)
    liks = vmap(lik_fn)(jnp.arange(nmix)) # liks(k,n)=p(X(n,:) | y=k)
    joint = jnp.einsum('kn,k -> nk', liks, prior) # joint(n,k) = liks(k,n) * prior(k)
    norm = joint.sum(axis=1) # norm(n)  = sum_k joint(n,k) = p(X(n,:)
    post = joint / jnp.expand_dims(norm, axis=1) # post(n,k) = p(y = k | xn)
    return post

class LabelSpace:
    def __init__(self, nclasses, nfactors):
        self.nclasses = nclasses
        self.nfactors = nfactors

    def flatten_labels(self, y, a):
        # to enable jit compilation, we set mode=clip
        #https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ravel_multi_index.html#jax.numpy.ravel_multi_index
        m = jnp.ravel_multi_index((y, a), (self.nclasses, self.nfactors), mode='clip')
        return m

    def unflatten_labels(self, m):
        ya = jnp.unravel_index(m, (self.nclasses, self.nfactors))
        return ya[0], ya[1]

    def unflatten_dist(self, z_dist):
        # unflatten distribution over z into labels y and attributes a
        if z_dist.ndim == 1: # single distribution (row vector, eg prior)
            reshaped = True
            n = len(z_dist)
            z_dist = jnp.reshape(z_dist, (1, n))
        else:
            reshaped = False

        z_dist = einops.rearrange(z_dist, 'n (y a) -> n y a', y=self.nclasses, a=self.nfactors)
        y_dist = einops.reduce(z_dist, 'n y a -> n y', 'sum') 
        a_dist = einops.reduce(z_dist, 'n y a -> n a', 'sum') 

        if reshaped:
            y_dist, a_dist = y_dist[0], a_dist[0]
        return y_dist, a_dist

def test_label_space():
    ls = LabelSpace(nclasses = 2, nfactors = 2)
    y = jnp.array([0, 0, 1, 1])
    a = jnp.array([0, 1, 0, 1])
    m = ls.flatten_labels(y,a)
    assert np.allclose(m, jnp.array([0,1,2,3]))
    yy, aa = ls.unflatten_labels(m)
    assert all(yy==y)
    assert all(aa==a)

