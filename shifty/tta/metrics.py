

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

def misclassification_rate(Ytrue, Yprob):
  Yhat = jnp.argmax(Yprob, axis=1)
  nerrors = jnp.sum(Yhat != Ytrue)
  nsamples = len(Ytrue)
  return nerrors / nsamples

#def mean_squared_error(class_post_true, class_post_pred):
#  return jnp.mean(jnp.power(class_post_true[:,0] - class_post_pred[:,0], 2))

def mean_squared_error(u, v):
  return jnp.mean(jnp.power(u - v, 2))

def true_false_positive(y_pred, y_test):
    true_positive = jnp.equal(y_pred, 1) & jnp.equal(y_test, 1)
    true_negative = jnp.equal(y_pred, 0) & jnp.equal(y_test, 0)
    false_positive = jnp.equal(y_pred, 1) & jnp.equal(y_test, 0)
    false_negative = jnp.equal(y_pred, 0) & jnp.equal(y_test, 1)
    tpr = true_positive.sum() / (true_positive.sum() + false_negative.sum())
    fpr = false_positive.sum() / (false_positive.sum() + true_negative.sum())
    return tpr, fpr

def roc(y_test, probabilities):
    ''' Same as sklearn.metrics.roc_curve '''
    # set thresholds big to small, so that tpr gradually increases
    thresholds = jnp.linspace(0.99, 0.01, num=200)
    #probs = jnp.unique(probabilities) # variable-sized, does not jit compile
    #thresholds = jnp.concatenate([jnp.array([0.01]), probs, jnp.array([0.99])])
    #thresholds = jnp.flip(thresholds)
    def f(thresh):
        threshold_vector = jnp.greater_equal(probabilities, thresh).astype(int)
        tpr, fpr = true_false_positive(threshold_vector, y_test)
        return jnp.array([fpr, tpr])        
    rocs =  vmap(f)(thresholds)
    return rocs[:,0], rocs[:,1], thresholds

def roc_auc(y_test, probabilities):
    ''' Same as sklearn.metrics.roc_auc_score '''
    fpr, tpr, thresholds = roc(y_test, probabilities)
    # https://github.com/akshaykapoor347/Compute-AUC-ROC-from-scratch-python/blob/master/AUCROCPython.ipynb
    return jnp.trapz(tpr, fpr)



