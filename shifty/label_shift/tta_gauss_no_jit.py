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
import typing

from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN

# Model from NURD paper:
# A. M. Puli, L. H. Zhang, E. K. Oermann, and R. Ranganath, 
# “Out-of-distribution Generalization in the Presence of Nuisance-Induced Spurious Correlations,” 
#  ICLR, May 2022 [Online]. Available: https://openreview.net/forum?id=12RoR2o32T.


@chex.dataclass
class NurdParams:
    b: float = 1 # strength of dependence of x on z (nuisance factor)
    sf: float = 1 # scale factor for Gaussian variance
    #generator: str = "nurd"

@chex.dataclass
class DiscrimParams:
    poly_degree: int = 2
    max_iter: int = 100

@chex.dataclass
class MetaParams: # these determine the sizeshape of the model
    ndim: float = 2 # dimensionality of generated X data
    nclasses: int = 2
    nfactors: int = 2 # other attributes
    nmix: int = 4 # nclasses * nfactors

@partial(jax.jit)
def class_cond_params_nurd(z, nurd_params, meta_params):  
    # returns parameters for distribution p(x|m=(y,a))
    ya = jnp.unravel_index(z, (meta_params.nclasses, meta_params.nfactors))
    y, a = ya[0], ya[1]
    b, sf = nurd_params.b, nurd_params.sf
    ysigned, asigned = 2.0*y-1,  2.0*a-1 #   # convert from (0,1) to (-1,1)
    mu = jnp.array([ysigned - b*asigned, ysigned + b*asigned])
    Sigma = sf*jnp.diag(jnp.array([1.5, 0.5]))
    return mu, Sigma

def lik_fn_nurd(z, X, nurd_params, meta_params):
    # returns p(X(n,:) | z) as (N,1) vector
    mu, Sigma = class_cond_params_nurd(z, nurd_params, meta_params)
    return jsp.stats.multivariate_normal.pdf(X, mu, Sigma)

#@partial(jax.jit, static_argnames=["meta_params"])
def sample_data_nurd(key, nsamples, prior, nurd_params, meta_params):
    labels = jr.categorical(key, logits=jnp.log(prior), shape=(nsamples,))
    def f(z):
        return class_cond_params_nurd(z, nurd_params, meta_params)
    mus, Sigmas = vmap(f)(jnp.arange(meta_params.nmix))
    X = jr.multivariate_normal(key, mus[labels], Sigmas[labels])
    return X, labels

def data_generator_nurd(key, n, prior, nurd_params, meta_params):
    XX, ZZ = sample_data_nurd(key, n, prior, nurd_params, meta_params)
    def lik_fn(z, X):
        return lik_fn_nurd(z, X, nurd_params, meta_params)
    ypost, zpost = predict_bayes(prior, lik_fn, XX, meta_params)
    return XX, ZZ, ypost

def make_prior(rho, meta_params):
    p_class =  jnp.array([0.5, 0.5]) # uniform prior  on p(y) = (-1, +1)
    p_factor_given_class = jnp.array([ [1-rho, rho], [rho, 1-rho]])
    pmix = jnp.einsum('y,ya->ya', p_class, p_factor_given_class) # pmiz(y,a)=p(y)*p(a|y)
    pmix = einops.rearrange(pmix, 'y a -> (y a)')
    return pmix

def compute_class_post_from_zpost(z_post, meta_params):
    z_post = einops.rearrange(z_post, 'n (y a) -> n y a', y=meta_params.nclasses, a=meta_params.nfactors)
    class_post = einops.reduce(z_post, 'n y a -> n y', 'sum') 
    return class_post

def predict_bayes(prior, lik_fn, X, meta_params):  
    liks = vmap(partial(lik_fn, X=X))(jnp.arange(meta_params.nmix)) # (K,N)
    joint = jnp.einsum('kn,k -> nk', liks, prior) # joint(n,k) = liks(k,n) * prior(k)
    norm = joint.sum(axis=1)
    zpost = joint / jnp.expand_dims(norm, axis=1) # joint_post(n,k) = p(z) = k | xn)
    ypost = compute_class_post_from_zpost(zpost, meta_params)
    return ypost, zpost



def fit_classifier(key, X, Z, meta_params):
    classifier = Pipeline([
            ('standardscaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2)), 
            ('logreg', LogisticRegression(random_state=0, max_iter=100))])
    classifier.fit(np.array(X), np.array(Z))
    return classifier

def predict_classifier(classifier, X, meta_params):
    zpost = jnp.array(classifier.predict_proba(X)) # (N,Z)
    ypost = compute_class_post_from_zpost(zpost, meta_params)
    return ypost, zpost

def classifier_to_lik_fn(classifier, prior, meta_params):
    def lik_fn(z, X):
        # return p_s(x(n) | z) = p_s(z|x) p_s(x) / p_s(z) propto p_s(z|x) / p_s(z)
        Yprobs, Zprobs = predict_classifier(classifier, X, meta_params)
        probs = Zprobs[:,z] / prior[z]
        return probs
    return lik_fn

def em(X, init_dist, lik_fn, meta_params):
  target_dist = init_dist
  pseudo_counts = 0.01 * init_dist
  niter = 5
  for t in range(niter):
    # E step
    ypost, zpost = predict_bayes(target_dist, lik_fn, X, meta_params) # zpost(n,k) = p(zn=k) 
    # M step
    counts = zpost + jnp.reshape(pseudo_counts, (1, meta_params.nmix)) 
    counts = einops.reduce(zpost, 'n k -> k', 'sum') # counts(k) = sum_n (zpost(n,k)+pseudo(k))
    target_dist = counts / jnp.sum(counts)
  return target_dist

def mse(u, v):
  return jnp.mean(jnp.power(u-v, 2))


@chex.dataclass
class SourceModel:
    prior: typing.Any 
    classifier: typing.Any 

def target_predict_unadapted(key, X, source_model, prior_target, meta_params):
    ypost, zpost =  predict_classifier(source_model.classifier, X, meta_params)
    return ypost

def target_predict_bayes(key, X, source_model, target_prior, meta_params):
    lik_fn = classifier_to_lik_fn(source_model.classifier, source_model.prior, meta_params)
    ypost, zpost = predict_bayes(target_prior, lik_fn, X, meta_params)
    return ypost

def target_fit_unadapted(key, X, source_model, meta_params):
    return source_model.prior

def target_fit_em(key, X, source_model, meta_params):
    lik_fn = classifier_to_lik_fn(source_model.classifier, source_model.prior, meta_params)
    target_prior = em(X, source_model.prior, lik_fn, meta_params)
    return target_prior

def source_fit_classifier(key, X, Z, meta_params):
    classifier = fit_classifier(key, X, Z, meta_params)
    probsY, probsZ = predict_classifier(classifier, X, meta_params)
    priorZ = jnp.mean(probsZ, axis=0) # prior(mz = empirical fraction of times z is predicted
    return SourceModel(classifier=classifier, prior=priorZ)


def eval_single_target(key_base, corr_target, data_generator_fn, source_model,
                        target_fit_fn, target_predict_fn, meta_params):
    prior_target_true = make_prior(corr_target, meta_params)
    nsamples_target = 500
    key, key_base = jr.split(key_base, 2)
    X, Z, Yprobs_true = data_generator_fn(key, nsamples_target, prior_target_true)
    prior_target_est = target_fit_fn(key, X, source_model, meta_params)
    Yprobs_pred = target_predict_fn(key, X, source_model, prior_target_est, meta_params)
    mse_probs = mse(Yprobs_true[:,0], Yprobs_pred[:,0])
    mse_prior = mse(prior_target_true, prior_target_est)
    return jnp.array([mse_probs, mse_prior])

def eval_multi_target(key_base, corr_source, corr_targets, data_generator_fn, source_fit_fn, target_fit_fn, target_predict_fn, meta_params):
    prior_source = make_prior(corr_source, meta_params)
    nsamples_source = 500
    key, key_base = jr.split(key_base, 2)
    X, Z, _ = data_generator_fn(key, nsamples_source, prior_source)
    key, key_base = jr.split(key_base, 2)
    source_model = source_fit_fn(key, X, Z, meta_params)
    key, key_base = jr.split(key_base, 2)
    def f(corr_target):
        return eval_single_target(key, corr_target, data_generator_fn, source_model, target_fit_fn, target_predict_fn, meta_params)
    losses = vmap(f)(corr_targets)
    #losses = []
    #for i, corr_target in enumerate(corr_targets):
    #    loss = f(corr_target)
    #    print(loss)
    #    losses.append(loss)

    return losses
