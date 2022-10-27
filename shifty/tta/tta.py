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
from shifty.tta.estimators import *
from shifty.skax.skax import *

def run():
    key = jr.PRNGKey(42)
    ls = LabelSpace(nclasses=2, nfactors=2)
    sf = 3
    def make_dist(key, rho):
        src_dist = NurdDataGenerator(key, rho, ls, sf=sf)
        return src_dist
    corr_sources = jnp.array([0.2, 0.5, 0.9])
    corr_targets = np.linspace(0.05, 0.95, num=10)
    ntrials = 5
    nsource_samples = 500
    ntarget_samples = 100
    clf = make_logreg


    methods = {
    'oracle': OracleEstimator(ls),
    'oracle-prior': OraclePriorEstimator(clf, ls),
    'unadapted': UndaptedEstimator(clf, ls),
    'em': EMEstimator(clf, ls)
    }

    losses_mean = {}
    losses_std = {}
    for name, estimator in methods.items():
        print(name)
        losses_mean[name], losses_std[name] = evaluate_estimator(key, make_dist,
            corr_sources, corr_targets, ntrials, estimator, nsource_samples = 500, ntarget_samples=100)

    print(losses_mean['unadapted'].shape) # (nsources, ntargets, nmetrics)

    nsources = len(corr_sources)
    metric_names = ['misclassification', 'mse', '1-auc']
    nmetrics = len(metric_names)
    fig, axs = plt.subplots(nmetrics, nsources, figsize=(20,10))
    fig.tight_layout(pad=3.0)
    for m in range(nmetrics):
        for s in range(nsources):
            ax = axs[m,s]
            corr_source = corr_sources[s]
            for name in methods.keys():
                ax.errorbar(corr_targets, losses_mean[name][s,:,m], yerr=losses_std[name][s,:,m], marker='o', label=name)
            ax.set_xlabel('correlation')
            ax.set_ylabel(metric_names[m])
            ax.set_title(r'Source sf={:0.1f}, $\rho={:0.1f}$'.format(sf, corr_source))
            ax.legend()
            ax.axvline(x=corr_source)


def run_nurd(corr_source=0.3, ntrials=3, sf=3):
    corr_targets = np.linspace(0.1, 0.9, num=9)

    key = jr.PRNGKey(420)
    keys = jr.split(key, ntrials)

    ls = LabelSpace(nclasses=2, nfactors=2)
    src_dist = NurdDataGenerator(key, corr_source, ls, sf=sf)
    def make_dist(key, rho):
        src_dist = NurdDataGenerator(key, rho, ls, sf=sf)

    clf = make_logreg()
    #clf = make_mlp()
    methods = {
    'oracle': OracleEstimator(ls),
    'oracle-prior': OraclePriorEstimator(clf, ls),
    'unadapted': UndaptedEstimator(clf, ls),
    'em': EMEstimator(clf, ls)
    }

    losses_mean = {}
    losses_std = {}
    for name, estimator in methods.items():
        print(name)
        losses_mean[name], losses_std[name] = evaluate_estimator_multi_trial(keys,
            src_dist, corr_targets, estimator, nsource_samples = 500, ntarget_samples=100)


    plt.figure()
    for name in methods.keys():
        plt.errorbar(corr_targets, losses_mean[name], yerr=losses_std[name], marker='o', label=name)
    plt.xlabel('correlation')
    plt.ylabel('Loss')
    plt.title(r'Source $\rho={:0.1f}$'.format(corr_source))
    plt.legend()
    plt.axvline(x=corr_source);

def debug_auc():
    key = jr.PRNGKey(42)
    corr_source = 0.3
    ls = LabelSpace(nclasses=2, nfactors=2)
    src_dist = PlantDataGenerator(key, corr_source, ls, sf=5)
    Xs, Ys, As = src_dist.sample(key, 500)
    clf = make_logreg
    estimator = UndaptedEstimator(clf, ls)
    #estimator = EMEstimator(clf, ls)

    estimator.fit_source(Xs, Ys, As, src_dist)

    corr = 0.9
    target_dist = deepcopy(src_dist)
    target_dist.shift_prior_correlation(corr)
    Xt, Yt, At = target_dist.sample(key, 100)
    true_prob = target_dist.predict_class(Xt)
    estimator.fit_target(Xt, target_dist)
    pred_prob = estimator.predict_class_target(Xt)

    y_test  = Yt
    prob_vector = pred_prob[:,1]


    fpr_sklearn, tpr_sklearn, thresholds_sklearn = sklearn.metrics.roc_curve(y_test, prob_vector)
    auc_sklearn = sklearn.metrics.roc_auc_score(y_test, prob_vector)
    fpr_kpm, tpr_kpm, thresholds_kpm = roc(y_test, prob_vector)
    auc_kpm = roc_auc(y_test, prob_vector)


    plt.figure()
    plt.scatter(fpr_sklearn, tpr_sklearn, s=100, alpha=0.5, color="blue", label="Scikit-learn")
    plt.scatter(fpr_kpm, tpr_kpm, color="red", s=100, alpha=0.3, label="Our implementation")
    plt.title("ROC Curve: AUC sklearn {:.3f}, us {:.3f}".format(auc_sklearn, auc_kpm))
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()