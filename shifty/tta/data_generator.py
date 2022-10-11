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

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN

from shifty.tta.label_space import *

def make_correlated_prior(rho, p_class=None):
    if p_class is None:
        p_class =  jnp.array([0.5, 0.5]) # uniform prior  on p(y) = (-1, +1)
    p_factor_given_class = jnp.array([ [1-rho, rho], [rho, 1-rho]])
    pmix = jnp.einsum('y,ya->ya', p_class, p_factor_given_class) # pmiz(y,a)=p(y)*p(a|y)
    pmix = einops.rearrange(pmix, 'y a -> (y a)')
    return pmix

@chex.dataclass
class GMMParams:
    nmix: int
    nfeatures: int
    prior: chex.Array
    mus: chex.Array # (C,D)
    Sigmas: chex.Array #(C,D,D)

def make_gmm_params(key, nmix, nfeatures, prior=None, scale_factor=1):
    # Random GMMs
    mus = jr.normal(key, (nmix, nfeatures)) # (C,D)
    # shared covariance -> linearly separable
    #Sigma = scale_factor * jnp.eye(nfeatures)
    #Sigmas = jnp.array([Sigma for _ in range(nmix)]) # (C,D,D)
    # diagonal covariance -> nonlinear decision boundaries
    sigmas = jr.uniform(key, shape=(nmix, nfeatures), minval=0.5, maxval=5)
    Sigmas = jnp.array([scale_factor*jnp.diag(sigmas[y]) for y in range(nmix)])
    if prior is None:
        prior = jnp.ones(nmix)/nmix
    return GMMParams(nmix=nmix, nfeatures=nfeatures, prior=prior, mus=mus, Sigmas=Sigmas)



class GMMDataGenerator:
    def __init__(self, key, correlation, label_space, nfeatures, class_prior=None, noise_factor=1):
        self.correlation = correlation
        self.label_space = label_space
        self.nfeatures = nfeatures
        self.nclasses = label_space.nclasses
        self.nfactors = label_space.nfactors
        self.nmix = self.nclasses * self.nfactors
        self.noise_factor = noise_factor
        if class_prior is None:
            self.class_prior = jnp.ones(self.nclasses)/self.nclasses
        else:
            self.class_prior = class_prior
        flat_prior = make_correlated_prior(self.correlation, self.class_prior)
        self.params = make_gmm_params(key, self.nmix, nfeatures, flat_prior, self.noise_factor)

    def sample(self, key, nsamples):
        z = jr.categorical(key, logits=jnp.log(self.params.prior), shape=(nsamples,))
        X = jr.multivariate_normal(key, self.params.mus[z], self.params.Sigmas[z])
        y, a = self.label_space.unflatten_labels(z)
        return X, y, a

    def shift_prior_correlation(self, correlation):
        self.correlation = correlation
        flat_prior = make_correlated_prior(self.correlation, self.class_prior)
        self.params.prior = flat_prior # just change prior of GMM

    def lik_fn(self, z, X):
        return   jsp.stats.multivariate_normal.pdf(X, self.params.mus[z], self.params.Sigmas[z])


def test_gmm():
    key = jr.PRNGKey(0)
    class_prior = jnp.array([0.3, 0.7])
    gmm_source = GMMDataGenerator(key, correlation=0.1, nclasses=2, nfactors=2, nfeatures=10, class_prior=class_prior)
    gmm_target = deepcopy(gmm_source)
    gmm_target.shift_prior_correlation(0.9)
    prior_y, prior_a = gmm_source.label_space.unflatten_dist(gmm_source.params.prior)
    assert np.allclose(prior_y, class_prior)
    prior_y, prior_a = gmm_target.label_space.unflatten_dist(gmm_target.params.prior)
    assert np.allclose(prior_y, class_prior)



def test_gmm2():
    key = jr.PRNGKey(0)
    class_prior = jnp.array([0.3, 0.7])
    gmm = GMMDataGenerator(key, correlation=0.1, nclasses=2, nfactors=2, nfeatures=10, class_prior=class_prior)
    prior_y, prior_a = gmm.label_space.unflatten_dist(gmm.params.prior)
    assert np.allclose(prior_y, class_prior)

    n = 1000
    X, y, a = gmm.sample(key, n)
    counts = jnp.sum(y==1); prior_y_mle = counts/n;
    counts = jnp.sum(a==1); prior_a_mle = counts/n;

    assert np.allclose(prior_y_mle, prior_y[1], atol=1e-1)
    assert np.allclose(prior_a_mle, prior_a[1], atol=1e-1)

# Model from NURD paper:
# A. M. Puli, L. H. Zhang, E. K. Oermann, and R. Ranganath, 
# “Out-of-distribution Generalization in the Presence of Nuisance-Induced Spurious Correlations,” 
#  ICLR, May 2022 [Online]. Available: https://openreview.net/forum?id=12RoR2o32T.


def class_cond_params_nurd(z, nclasses, nfactors, b, sf):  
    # returns parameters for distribution p(x|m=(y,a))
    ya = jnp.unravel_index(z, (nclasses, nfactors))
    y, a = ya[0], ya[1]
    ysigned, asigned = 2.0*y-1,  2.0*a-1 #   # convert from (0,1) to (-1,1)
    mu = jnp.array([ysigned - b*asigned, ysigned + b*asigned])
    Sigma = sf*jnp.diag(jnp.array([1.5, 0.5]))
    return mu, Sigma

class NurdDataGenerator(GMMDataGenerator):
    def __init__(self, correlation, b=1, sf=1):
        self.correlation = correlation
        key = jr.PRNGKey(0)
        nclasses, nfactors, nmix = 2, 2, 4
        label_space = LabelSpace(nclasses = nclasses, nfactors = nfactors)
        self.class_prior = jnp.array([0.5,0.5])
        nfeatures = 2
        super().__init__(key, correlation, label_space, nfeatures, class_prior=self.class_prior, noise_factor=1)
        def f(z):
            return class_cond_params_nurd(z, nclasses, nfactors, b, sf)
        mus, Sigmas = vmap(f)(jnp.arange(nmix))
        flat_prior = make_correlated_prior(self.correlation, self.class_prior)
        self.params = GMMParams(nmix=nmix, nfeatures=nfeatures, prior=flat_prior, mus=mus, Sigmas=Sigmas)


