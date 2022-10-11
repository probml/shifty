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
from shifty.tta.estimators import *
from shifty.tta.data_generator import *
from shifty.skax.skax import *

def evaluate_predictions(Ytrue, Yprob):
  Yhat = jnp.argmax(Yprob, axis=1)
  nerrors = jnp.sum(Yhat != Ytrue)
  nsamples = len(Ytrue)
  return nerrors / nsamples

def evaluate_estimator(key, src_dist, corr_targets, estimator, nsource_samples = 500, ntarget_samples=100):
    Xs, Ys, As = src_dist.sample(key, nsource_samples)
    estimator.fit_source(Xs, Ys, As, src_dist)
    ntargets = len(corr_targets)
    keys = jr.split(key, ntargets)
    def f(key_corr):
        key, corr = key_corr
        target_dist = deepcopy(src_dist)
        target_dist.shift_prior_correlation(corr)
        Xt, Yt, At = target_dist.sample(key, ntarget_samples)
        estimator.fit_target(Xt, target_dist)
        Ypred = estimator.predict_target(Xt)
        return evaluate_predictions(Yt, Ypred)
    metrics = vmap(f)((keys, corr_targets))
    return metrics

        
def demo():
    key = jr.PRNGKey(0)
    corr_source = 0.1
    corr_targets = jnp.arange(0.1, 0.9, step=0.1)
    ls = LabelSpace(nclasses=2, nfactors=2)
    src_dist = GMMDataGenerator(key, corr_source, ls, nfeatures=4)

    nhidden = (10,) + (ls.nclasses,) # set nhidden = () + (ls.classes,) to get logistic regression
    network = MLPNetwork(nhidden)
    mlp = NeuralNetClassifier(network, key, ls.nclasses, l2reg=1e-5, optimizer = "adam+warmup", 
            batch_size=32, num_epochs=20, print_every=0) 

    est = OracleEstimator(ls)
    metrics_oracle = evaluate_estimator(key, src_dist, corr_targets, est)

    est = UndaptedEstimator(mlp, ls)
    metrics_unadapted = evaluate_estimator(key, src_dist, corr_targets, est)

    est = EMEstimator(mlp, ls)
    metrics_em = evaluate_estimator(key, src_dist, corr_targets, est)


    plt.figure;
    plt.plot(corr_targets, metrics_oracle, 'b--', label='oracle')
    plt.plot(corr_targets, metrics_unadapted, 'k:', label='unadapted')
    plt.plot(corr_targets, metrics_em, 'r-', label='em')
    plt.xlabel('correlation')
    plt.ylabel('performance')
