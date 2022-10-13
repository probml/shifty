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

def run_expt(corr_source=0.3, ntrials=3, sf=3):
    corr_targets = np.linspace(0.1, 0.9, num=9)

    key = jr.PRNGKey(420)
    keys = jr.split(key, ntrials)

    ls = LabelSpace(nclasses=2, nfactors=2)
    src_dist = NurdDataGenerator(key, corr_source, ls, sf=sf)
    def make_dist(rho):
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
