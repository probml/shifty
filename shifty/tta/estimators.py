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

def predict_bayes(prior, likelihood_fn, X):
    nmix = len(prior)
    lik_fn = partial(likelihood_fn, X=X)
    liks = vmap(lik_fn)(jnp.arange(nmix)) # liks(k,n)=p(X(n,:) | y=k)
    joint = jnp.einsum('kn,k -> nk', liks, prior) # joint(n,k) = liks(k,n) * prior(k)
    norm = joint.sum(axis=1) # norm(n)  = sum_k joint(n,k) = p(X(n,:)
    post = joint / jnp.expand_dims(norm, axis=1) # post(n,k) = p(y = k | xn)
    return post



class OracleEstimator:
    def __init__(self, label_space):
        self.prior_source = None
        self.prior_target = None
        self.label_space = label_space
        self.lik_fn = None
        
    def fit_source(self, Xs, Ys, As, src_dist):
        del Xs, Ys, As
        self.prior_source = src_dist.params.prior
        self.lik_fn = src_dist.lik_fn
    
    def fit_target(self, X, target_dist):
        self.prior_target = target_dist.params.prior

    def predict_source(self, X):
        zpost = predict_bayes(self.prior_source, self.lik_fn, X)
        ypost, apost = self.label_space.unflatten_dist(zpost)
        return ypost 

    def predict_target(self, X):
        zpost = predict_bayes(self.prior_target, self.lik_fn, X)
        ypost, apost = self.label_space.unflatten_dist(zpost)
        return ypost 

def em(init_dist, lik_fn, X, niter):
    target_dist = init_dist
    pseudo_counts = 0.01 * init_dist
    nmix = len(init_dist)
    for t in range(niter):
        # E step
        zpost = predict_bayes(target_dist, lik_fn, X) # zpost(n,k) = p(zn=k) 
        # M step
        counts = zpost + jnp.reshape(pseudo_counts, (1, nmix)) 
        counts = einops.reduce(zpost, 'n k -> k', 'sum') # counts(k) = sum_n (zpost(n,k)+pseudo(k))
        target_dist = counts / jnp.sum(counts)
    return target_dist


def classifier_to_lik_fn(classifier, prior):
    def lik_fn(z, X):
        # return p_s(x | z) = p_s(z|x) p_s(x) / p_s(z) propto p_s(z|x) / p_s(z)
        Zprobs = classifier.predict(X) #(N,Z)
        probs = Zprobs[:,z] / prior[z]
        return probs
    return lik_fn

class EMEstimator:
    def __init__(self, classifier, label_space, num_em_iter=5):
        self.classifier = deepcopy(classifier)
        self.prior_source = []
        self.prior_target = []
        self.label_space = label_space
        self.lik_fn = []
        self.num_em_iter = num_em_iter

    def fit_source(self, Xs, Ys, As, src_dist):
        del src_dist
        Zs = self.label_space.flatten_labels(Ys, As)
        self.classifier.fit(Xs, Zs)
        probsZ = self.classifier.predict(Xs)
        self.prior_source = jnp.mean(probsZ, axis=0) #  empirical fraction of times z is predicted
        self.lik_fn = classifier_to_lik_fn(self.classifier, self.prior_source)

    def fit_target(self, X, target_dist):
        del target_dist
        self.prior_target = em(self.prior_source, self.lik_fn, X, self.num_em_iter)

    def predict_source(self, X):
        zpost = self.classifier.predict(X)
        ypost, apost = self.label_space.unflatten_dist(zpost)
        return ypost 

    def predict_target(self, X):
        zpost = predict_bayes(self.prior_target, self.lik_fn, X)
        ypost, apost = self.label_space.unflatten_dist(zpost)
        return ypost 


class UndaptedEstimator:
    def __init__(self, classifier, label_space):
        self.classifier = deepcopy(classifier)
        self.label_space = label_space

    def fit_source(self, Xs, Ys, As, src_dist):
        del src_dist
        Zs = self.label_space.flatten_labels(Ys, As)
        self.classifier.fit(Xs, Zs)
    
    def fit_target(self, X, target_dist):
        del target_dist
        pass

    def predict_source(self, X):
        zpost = self.classifier.predict(X)
        ypost, apost = self.label_space.unflatten_dist(zpost)
        return ypost 

    def predict_target(self, X):
        return self.predict_source(X)
