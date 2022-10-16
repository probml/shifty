from einops.einops import ParsedExpression
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

from shifty.tta.label_space import *
from shifty.tta.data_generator import *
from shifty.tta.metrics import *
from shifty.skax.skax import *

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
        del X
        self.prior_target = target_dist.params.prior

    def predict_joint_source(self, X):
        return predict_bayes(self.prior_source, self.lik_fn, X)

    def predict_class_source(self, X):
        zpost = self.predict_joint_source(X)
        ypost, apost = self.label_space.unflatten_dist(zpost)
        return ypost 

    def predict_joint_target(self, X):
        return predict_bayes(self.prior_target, self.lik_fn, X)

    def predict_class_target(self, X):
        zpost = self.predict_joint_target(X)
        ypost, apost = self.label_space.unflatten_dist(zpost)
        return ypost 



def em(X, init_dist, lik_fn, niter, pseudo_counts):
    target_dist = init_dist
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
    def __init__(self, classifier, label_space, num_em_iter=5, prior_strength=0.01):
        self.classifier = deepcopy(classifier)
        self.prior_source = []
        self.prior_target = []
        self.label_space = label_space
        self.lik_fn = []
        self.num_em_iter = num_em_iter
        self.prior_strength = prior_strength

    def fit_source(self, Xs, Ys, As, src_dist):
        del src_dist
        Zs = self.label_space.flatten_labels(Ys, As)
        self.classifier.fit(Xs, Zs)
        probsZ = self.classifier.predict(Xs)
        self.prior_source = jnp.mean(probsZ, axis=0) #  empirical fraction of times z is predicted
        self.lik_fn = classifier_to_lik_fn(self.classifier, self.prior_source)

    def fit_target(self, X, target_dist):
        del target_dist
        nmix = len(self.prior_source)
        self.prior_target = em(X, self.prior_source, self.lik_fn, 
            self.num_em_iter, self.prior_strength*jnp.ones(nmix))
    
    def predict_joint_source(self, X):
        #return predict_bayes(self.prior_source, self.lik_fn, X)
        return self.classifier.predict(X)

    def predict_class_source(self, X):
        zpost = self.predict_joint_source(X)
        ypost, apost = self.label_space.unflatten_dist(zpost)
        return ypost 

    def predict_joint_target(self, X):
        return predict_bayes(self.prior_target, self.lik_fn, X)

    def predict_class_target(self, X):
        zpost = self.predict_joint_target(X)
        ypost, apost = self.label_space.unflatten_dist(zpost)
        return ypost

class OraclePriorEstimator(EMEstimator):
    def __init__(self, classifier, label_space):
        super().__init__(classifier, label_space, num_em_iter=None)

    def fit_target(self, X, target_dist):
        del X
        self.prior_target = target_dist.params.prior


class UndaptedEstimator:
    def __init__(self, classifier, label_space):
        self.classifier = deepcopy(classifier)
        self.label_space = label_space

    def fit_source(self, Xs, Ys, As, src_dist):
        del src_dist
        Zs = self.label_space.flatten_labels(Ys, As)
        self.classifier.fit(Xs, Zs)
    
    def fit_target(self, X, target_dist):
        pass

    def predict_joint_source(self, X):
        return self.classifier.predict(X)

    def predict_class_source(self, X):
        zpost = self.predict_joint_source(X)
        ypost, apost = self.label_space.unflatten_dist(zpost)
        return ypost 

    def predict_joint_target(self, X):
        return self.predict_joint_source(X)

    def predict_class_target(self, X):
        return self.predict_class_source(X)



def evaluate_estimator_single_trial(key, src_dist, corr_targets, estimator, nsource_samples = 500, ntarget_samples=100):
    # compare on a range of target distributions
    Xs, Ys, As = src_dist.sample(key, nsource_samples)
    estimator.fit_source(Xs, Ys, As, src_dist)
    ntargets = len(corr_targets)
    @jit
    def f(corr):
        target_dist = deepcopy(src_dist)
        target_dist.shift_prior_correlation(corr)
        Xt, Yt, At = target_dist.sample(key, ntarget_samples)
        true_prob = target_dist.predict_class(Xt)
        estimator.fit_target(Xt, target_dist)
        pred_prob = estimator.predict_class_target(Xt)
        metric1 =  misclassification_rate(Yt, pred_prob)
        metric2 = mean_squared_error(true_prob, pred_prob)
        metric3 = 1-roc_auc(Yt, pred_prob[:,1])
        #metric4 = 1-roc_auc(Yt, pred_prob[:,0]) # wrong 
        metric = jnp.array([metric1, metric2, metric3])
        return metric
    metrics = vmap(f)(corr_targets) # (ntargets, nmetrics)
    return metrics


def evaluate_estimator_multi_trial(keys, src_dist, corr_targets, estimator, nsource_samples = 500, ntarget_samples=100):
    @jit
    def f(key):
        return evaluate_estimator_single_trial(key, src_dist, corr_targets, estimator, nsource_samples, ntarget_samples)
    losses_per_trial = vmap(f)(keys) # (ntrials, ntargets, nmetrics)
    losses_mean = jnp.mean(losses_per_trial, axis=0) # (ntargets, nmetrics)
    losses_std = jnp.std(losses_per_trial, axis=0)
    return losses_mean, losses_std

def evaluate_estimator_old(key, make_dist, corr_sources, corr_targets, ntrials, estimator, nsource_samples = 500, ntarget_samples=100):
    # use different key for every source
    @jit
    def f(key, rho):
        src_dist =  make_dist(key, rho)
        keys = jr.split(key, ntrials)
        losses_mean, losses_std = evaluate_estimator_multi_trial(keys, src_dist, corr_targets, estimator, nsource_samples, ntarget_samples)
        return losses_mean, losses_std # each is size (ntargets, nmetrics)
    keys = jr.split(key, len(corr_sources))
    losses_mean, losses_std = vmap(f)(keys, corr_sources) # (nsources, ntargets, nmetrics)
    return losses_mean, losses_std

def evaluate_estimator(key, make_dist, corr_sources, corr_targets, ntrials, estimator, nsource_samples = 500, ntarget_samples=100):
    # use same key for every source
    @jit
    def f(rho):
        src_dist =  make_dist(key, rho)
        keys = jr.split(key, ntrials)
        losses_mean, losses_std = evaluate_estimator_multi_trial(keys, src_dist, corr_targets, estimator, nsource_samples, ntarget_samples)
        return losses_mean, losses_std # each is size (ntargets, nmetrics)
    #keys = jr.split(key, len(corr_sources))
    losses_mean, losses_std = vmap(f)(corr_sources) # (nsources, ntargets, nmetrics)
    return losses_mean, losses_std

def make_mlp():
    key = jr.PRNGKey(42)
    ntargets = 4
    nhidden =  (10, 10, ntargets)
    network = MLPNetwork(nhidden)
    model = NeuralNetClassifier(network, key, ntargets, l2reg=1e-2, optimizer = "adam+warmup", 
            batch_size=32, num_epochs=20, print_every=0)
    return model

def make_logreg():
    key = jr.PRNGKey(42)
    ntargets = 4
    nhidden =  (ntargets,)
    network = MLPNetwork(nhidden)
    model = NeuralNetClassifier(network, key, ntargets,
        l2reg=1e-5, standardize=True, max_iter=500, optimizer = "lbfgs")
    return model