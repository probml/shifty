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

# Model from NURD paper:
# A. M. Puli, L. H. Zhang, E. K. Oermann, and R. Ranganath, 
# “Out-of-distribution Generalization in the Presence of Nuisance-Induced Spurious Correlations,” 
#  ICLR, May 2022 [Online]. Available: https://openreview.net/forum?id=12RoR2o32T.


@chex.dataclass
class NurdParams:
    b: float = 1 # strength of dependence of x on z (nuisance factor)
    sf: float = 1 # scale factor for Gaussian variance
    generator: str = "nurd"

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


def class_cond_params_nurd(m, nurd_params, meta_params):  
    # returns parameters for distribution p(x|m=(y,z))
    yz = jnp.unravel_index(m, (meta_params.nclasses, meta_params.nfactors))
    y, z = yz[0], yz[1]
    b, sf = nurd_params.b, nurd_params.sf
    ysigned, zsigned = 2.0*y-1,  2.0*z-1 #   # convert from (0,1) to (-1,1)
    mu = jnp.array([ysigned - b*zsigned, ysigned + b*zsigned])
    Sigma = sf*jnp.diag(jnp.array([1.5, 0.5]))
    return mu, Sigma

def lik_fn_nurd(m, X, nurd_params, meta_params):
    # returns p(X(n,:) | m) as (N,1) vector
    mu, Sigma = class_cond_params_nurd(m, nurd_params, meta_params)
    return jsp.stats.multivariate_normal.pdf(X, mu, Sigma)

def sample_data_nurd(key, nsamples, prior, nurd_params, meta_params):
    labels = jr.categorical(key, logits=jnp.log(prior), shape=(nsamples,))
    def f(m):
        return class_cond_params_nurd(m, nurd_params, meta_params)
    mus, Sigmas = vmap(f)(jnp.arange(meta_params.nmix))
    X = jr.multivariate_normal(key, mus[labels], Sigmas[labels])
    return X, labels


def make_prior(rho, meta_params):
    p_class =  np.array([0.5, 0.5]) # uniform prior  on p(y) = (-1, +1)
    p_factor_given_class = np.zeros((2, 2))  # p(z|c) = p_factor(c,z) so each row sums to 1
    p_factor_given_class[0, :] = [1 - rho, rho]
    p_factor_given_class[1, :] = [rho, 1 - rho]
    pmix = jnp.einsum('y,yz->yz', p_class, p_factor_given_class)
    pmix = einops.rearrange(pmix, 'y z -> (y z)')
    return pmix

def compute_class_post_from_joint(mix_post, meta_params):
    mix_post = einops.rearrange(mix_post, 'n (y z) -> n y z', y=meta_params.nclasses, z=meta_params.nfactors)
    class_post = einops.reduce(mix_post, 'n y z -> n y', 'sum') 
    return class_post

def predict_bayes(prior, lik_fn, X, meta_params):  
    liks = vmap(partial(lik_fn, X=X))(jnp.arange(meta_params.nmix)) # (K,N)
    joint = jnp.einsum('kn,k -> nk', liks, prior) # joint(n,k) = liks(k,n) * prior(k)
    norm = joint.sum(axis=1)
    joint_post = joint / jnp.expand_dims(norm, axis=1) # joint_post(n,k) = p(mix = k | xn)
    class_post = compute_class_post_from_joint(joint_post, meta_params)
    return class_post

def predict_source(classifier, X, meta_params):
    joint_post = classifier.predict_proba(X) 
    return compute_class_post_from_joint(joint_post, meta_params)

def fit_source(key, X, joint_labels, discrim_params):
    classifier = Pipeline([
            ('standardscaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=discrim_params.poly_degree)), 
            ('logreg', LogisticRegression(random_state=0, max_iter=discrim_params.max_iter))])
    classifier.fit(X, joint_labels)
    probs_joint = classifier.predict_proba(X) # (n,m)
    prior_joint = jnp.mean(probs_joint, axis=0) # prior(m) = empirical fraction of times m is predicted
    return classifier, prior_joint


def classifier_to_lik_fn(classifier, prior):
    def lik_fn(m, X):
        # return p_s(x(n) | m) = p_s(m|x) p_s(x) / p_s(m) propto p_s(m|x) / p_s(m)
        probs = jnp.array(classifier.predict_proba(X))
        probs = probs[:,m] / prior[m]
        return probs
    return lik_fn

def predict_target(prior, classifier, X, meta_params):
    lik_fn = classifier_to_lik_fn(classifier)
    return predict_bayes(prior, lik_fn, X, meta_params)



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

