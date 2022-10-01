import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import einops
import matplotlib
from functools import partial
from collections import namedtuple

import sklearn
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import LogisticRegression

import jax
import jax.random as jr
import jax.scipy as jsp 
import jax.numpy as jnp
from jax import vmap, grad, jit
import chex

from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN


NCLASSES = 2
NFACTORS = 2 # nuisance factors
NMIX = NCLASSES * NFACTORS # mixture components
NDIM = 2 # dimensionality of observed data

def yz_to_mix(y, z):
    m = jnp.ravel_multi_index((jnp.array([y]), jnp.array([z])), (NCLASSES, NFACTORS))
    return m[0]

def mix_to_yz(m):
    yz = jnp.unravel_index(m, (NCLASSES, NFACTORS))
    return yz[0], yz[1]

@chex.dataclass
class GenParams:
    b: float # strength of dependence of x on z (nuisance factor)
    sf: float # scale factor for Gaussian variance

def class_cond_density_nurd(y, z, b, sf):  
    # returns parameters for distribution p(x|y,z)
    # Model from NURD paper:
    # A. M. Puli, L. H. Zhang, E. K. Oermann, and R. Ranganath, 
    # “Out-of-distribution Generalization in the Presence of Nuisance-Induced Spurious Correlations,” 
    #  ICLR, May 2022 [Online]. Available: https://openreview.net/forum?id=12RoR2o32T. 
    ysigned, zsigned = 2.0*y-1,  2.0*z-1 #   # convert from (0,1) to (-1,1)
    mu = jnp.array([ysigned - b*zsigned, ysigned + b*zsigned])
    Sigma = sf*jnp.diag(jnp.array([1.5, 0.5]))
    return mu, Sigma

def make_lik_dist(m, gen_params): # deprecated
    y, z = mix_to_yz(m)
    mu, Sigma = class_cond_density_nurd(y, z, gen_params.b, gen_params.sf)
    dist = MVN(loc = mu, covariance_matrix = Sigma)
    return dist

def lik_fn_old(m, X, gen_params): # deprecated
    # returns p(X(n,:) | m) as (N,1) vector
    dist = make_lik_dist(m, gen_params)
    return dist.prob(X)

def lik_fn(m, X, gen_params):
    # returns p(X(n,:) | m) as (N,1) vector
    y, z = mix_to_yz(m)
    mu, Sigma = class_cond_density_nurd(y, z, gen_params.b, gen_params.sf)
    return jsp.multivariate_normal.prob(X, mu, Sigma)

def make_mix_dist(prior, gen_params): # deprecated
    def f(m):
        dist = make_lik_dist(m, gen_params)
        return dist.mean(), dist.covariance()
    mus, Sigmas = vmap(f)(jnp.arange(NMIX))
    dist = tfd.MixtureSameFamily(
      mixture_distribution=tfd.Categorical(probs=prior),
      components_distribution=MVN(loc = mus, covariance_matrix = Sigmas))
    joint_dist = tfd.JointDistributionSequential(
        [tfd.Categorical(logits=jnp.log(prior)), 
        lambda label: MVN(loc=mus[label], covariance_matrix=Sigmas[label])])
    return dist, joint_dist

def sample_data_old(key, prior, gen_params, nsamples): # deprecated
    labels =  jr.categorical(key, logits=jnp.log(prior), shape=(nsamples,))
    X = jnp.zeros((nsamples, NDIM))
    for m in range(NMIX):
        ndx = jnp.nonzero(labels==m)[0]
        n = len(ndx) # num. samples for this mixture component
        dist = make_lik_dist(m, gen_params)
        samples = dist.sample(seed=key, sample_shape=n)
        X = X.at[ndx].set(samples)
    return X, labels

def sample_data(key, prior, gen_params, nsamples):
    labels = jr.categorical(key, logits=jnp.log(prior), shape=(nsamples,))
    def f(m):
        y, z = mix_to_yz(m)
        mu, Sigma = class_cond_density_nurd(y, z, gen_params.b, gen_params.sf)
        return mu, Sigma
    mus, Sigmas = vmap(f)(jnp.arange(NMIX))
    X = jr.multivariate_normal(key, mus[labels], Sigmas[labels])
    return X, labels

def make_prior(rho):
    p_class =  np.array([0.5, 0.5]) # uniform prior  on p(y) = (-1, +1)
    p_factor_given_class = np.zeros((2, 2))  # p(z|c) = p_factor(c,z) so each row sums to 1
    p_factor_given_class[0, :] = [1 - rho, rho]
    p_factor_given_class[1, :] = [rho, 1 - rho]
    p_mix = np.zeros((NCLASSES, NFACTORS))  # (c,z)
    for c in range(NCLASSES):
        for z in range(NFACTORS):
            p_mix[c, z] = p_class[c] * p_factor_given_class[c, z]
    p_mix = einops.rearrange(p_mix, 'y z -> (y z)')
    return p_mix

def predict_bayes(prior, lik_fn, X, gen_params):  
    liks = vmap(partial(lik_fn, X=X, gen_params=gen_params))(jnp.arange(NMIX)) # (K,N)
    joint = jnp.einsum('kn,k -> nk', liks, prior)
    norm = joint.sum(axis=1)
    mix_post = joint / jnp.expand_dims(norm, axis=1)
    mix_post = einops.rearrange(mix_post, 'n (y z) -> n y z', y=NCLASSES, z=NFACTORS)
    class_post = einops.reduce(mix_post, 'n y z -> n y', 'sum') 
    return class_post


def predict_source(classifier, x):
    return classifier.proba(x)

def classifier_to_lik_fn(classifier):
    pass

def predict_target(prior, classifier, x):
    lik_fn = classifier_to_lik_fn(classifier)
    return predict_bayes(prior, lik_fn, x)

def fit_source(key, labeled_data):
    model = None
    return model

def fit_target(key, model, unlabeled_data):
    #model = update(model, unlabeled_data)
    return model


def evaluate_preds(class_post_true, class_post_pred):
  # we use mean squared error on class 0 (could also use AUC)
  return jnp.mean(jnp.power(class_post_true[:,0] - class_post_pred[:,0], 2))

def eval_single_target(key, corr_target, lik_fn, model):
    prior_target = make_prior(corr_target)
    nsamples_target = 100
    data_target = sample_data(key, prior_target, lik_fn, nsamples_target)
    model = fit_target(key, model, data_target)
    probs_pred = predict_target(model, data_target)
    probs_true = predict_bayes(prior_target, lik_fn, data_target)
    loss = evaluate_preds(probs_true, probs_pred)
    return loss

def eval_multi_target(key, corr_source, corr_targets, lik_fn):
    prior_source = make_prior(corr_source)
    nsamples_source = 500
    data_source = sample_data(key, prior_source, lik_fn, nsamples_source)
    model = fit_source(key, data_source)
    def f(corr_target):
        return eval_single_target(key, corr_target, lik_fn, model)
    losses = vmap(f)(corr_targets)
    return losses

