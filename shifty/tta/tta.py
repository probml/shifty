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

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN


nclasses = 2
nfactors = 2
nmix = 4

def make_prior(rho):
    p_class =  jnp.array([0.5, 0.5]) # uniform prior  on p(y) = (-1, +1)
    p_factor_given_class = jnp.array([ [1-rho, rho], [rho, 1-rho]])
    pmix = jnp.einsum('y,ya->ya', p_class, p_factor_given_class) # pmiz(y,a)=p(y)*p(a|y)
    pmix = einops.rearrange(pmix, 'y a -> (y a)')
    return pmix

@chex.dataclass
class GenParams:
    nclasses: int
    nfeatures: int
    prior: chex.Array
    mus: chex.Array # (C,D)
    Sigmas: chex.Array #(C,D,D)

def make_params(key, nclasses, nfeatures, prior=None, scale_factor=1):
    mus = jr.normal(key, (nclasses, nfeatures)) # (C,D)
    # shared covariance -> linearly separable
    #Sigma = scale_factor * jnp.eye(nfeatures)
    #Sigmas = jnp.array([Sigma for _ in range(nclasses)]) # (C,D,D)
    # diagonal covariance -> nonlinear decision boundaries
    sigmas = jr.uniform(key, shape=(nclasses, nfeatures), minval=0.5, maxval=5)
    Sigmas = jnp.array([scale_factor*jnp.diag(sigmas[y]) for y in range(nclasses)])
    if prior is None:
        prior = jnp.ones(nclasses)/nclasses
    return GenParams(nclasses=nclasses, nfeatures=nfeatures, prior=prior, mus=mus, Sigmas=Sigmas)

def sample_data(key, params, nsamples):
    y = jr.categorical(key, logits=jnp.log(params.prior), shape=(nsamples,))
    X = jr.multivariate_normal(key, params.mus[y], params.Sigmas[y])
    return X, y

def predict_bayes(X, params):
    def lik_fn(y):
        return   jsp.stats.multivariate_normal.pdf(X, params.mus[y], params.Sigmas[y])
    liks = vmap(lik_fn)(jnp.arange(params.nclasses)) # liks(k,n)=p(X(n,:) | y=k)
    joint = jnp.einsum('kn,k -> nk', liks, params.prior) # joint(n,k) = liks(k,n) * prior(k)
    norm = joint.sum(axis=1) # norm(n)  = sum_k joint(n,k) = p(X(n,:)
    post = joint / jnp.expand_dims(norm, axis=1) # post(n,k) = p(y = k | xn)
    return post


def compute_class_post_from_zpost(z_post):
    z_post = einops.rearrange(z_post, 'n (y a) -> n y a', y=nclasses, a=nfactors)
    class_post = einops.reduce(z_post, 'n y a -> n y', 'sum') 
    return class_post

def predict_classifier(classifier, X):
    zpost = classifier.predict(X) # (N,Z)
    ypost = compute_class_post_from_zpost(zpost)
    return ypost, zpost

def classifier_to_lik_fn(classifier, prior):
    def lik_fn(z, X):
        # return p_s(x(n) | z) = p_s(z|x) p_s(x) / p_s(z) propto p_s(z|x) / p_s(z)
        Yprobs, Zprobs = predict_classifier(classifier, X)
        probs = Zprobs[:,z] / prior[z]
        return probs
    return lik_fn

def em(X, init_dist, lik_fn):
  target_dist = init_dist
  pseudo_counts = 0.01 * init_dist
  niter = 5
  for t in range(niter):
    # E step
    ypost, zpost = predict_bayes(target_dist, lik_fn, X) # zpost(n,k) = p(zn=k) 
    # M step
    counts = zpost + jnp.reshape(pseudo_counts, (1, nmix)) 
    counts = einops.reduce(zpost, 'n k -> k', 'sum') # counts(k) = sum_n (zpost(n,k)+pseudo(k))
    target_dist = counts / jnp.sum(counts)
  return target_dist

@partial(jax.jit)
def mse(u, v):
  return jnp.mean(jnp.power(u-v, 2))


@partial(jax.jit)
def target_predict_unadapted(key, X, source_prior, classifier, target_prior):
    ypost, zpost =  predict_classifier(classifier, X)
    return ypost

@partial(jax.jit)
def target_predict_bayes(key, X, source_prior, classifier, target_prior):
    lik_fn = classifier_to_lik_fn(classifier, source_prior)
    ypost, zpost = predict_bayes(target_prior, lik_fn, X)
    return ypost

@partial(jax.jit, static_argnames=["classifier"])
def target_fit_unadapted(key, X, source_prior, classifier):
    return source_prior

@partial(jax.jit)
def target_fit_em(key, X, source_prior, classifier):
    lik_fn = classifier_to_lik_fn(classifier, source_prior)
    target_prior = em(X, source_prior, lik_fn)
    return target_prior


def eval_single_target(key_base, corr_target, classifier, target_fit_fn, source_prior_est, params_source):
    params_target = dataclasses.replace(params_source)
    params_target.prior = make_prior(corr_target)
    nsamples_target = 500
    key, subkey = jr.split(key, 2)
    X, Z = sample_data(subkey, params_target, nsamples_target)
    prior_target_est = target_fit_fn(key, X, source_prior_est, classifier)

    Yprobs_pred = target_predict_fn(key, X, source_prior, classifier, prior_target_est)
    mse_probs = mse(Yprobs_true[:,0], Yprobs_pred[:,0])
    mse_prior = mse(prior_target_true, prior_target_est)
    return jnp.array([mse_probs, mse_prior])

def eval_multi_target(key, corr_source, corr_targets, classifier, target_fit_fn):
    prior = make_prior(corr_source)
    nclasses = len(prior)
    key, subkey = jr.split(key, 2)
    gen_params = make_params(subkey, nclasses, nfeatures=10, prior=prior, scale_factor=1)
    nsamples_source = 500
    key, subkey = jr.split(key, 2)
    X, Z = sample_data(subkey, gen_params, nsamples_source)

    key, subkey = jr.split(key, 2)
    classifier.fit(subkey, X, Z)
    probsZ = classifier.predict(X)
    priorZ = jnp.mean(probsZ, axis=0) # prior(mz = empirical fraction of times z is predicted

    ntargets = len(corr_targets)
    keys = jr.split(key, ntargets)
    def f(key, corr_target):
        return eval_single_target(key, corr_target, classifier, target_fit_fn, priorZ, gen_params)
    losses = vmap(f)(zip(keys, corr_targets))
    return losses

'''
def demo():
    key = jr.PRNGKey(0)
    corr_source = 0.1; corr_targets = jnp.arange(0.1, 0.9, step=0.1)
    losses_unadapted = eval_multi_target(key, corr_source, corr_targets)
    print(losses_unadapted)

    source_fit_fn = source_fit_classifier
    target_fit_fn  = target_fit_em
    target_predict_fn = target_predict_bayes

    losses_em = eval_multi_target(key, corr_source, corr_targets,
        data_generator_fn, source_fit_fn, target_fit_fn, target_predict_fn, meta_params)
    print(losses_em)

    plt.figure;
    plt.plot(corr_targets, losses_em[:,0], 'r-', label='em')
    plt.plot(corr_targets, losses_unadapted[:,0], 'k:', label='unadapted')
    plt.xlabel('correlation')
    plt.ylabel('mse probs')
'''