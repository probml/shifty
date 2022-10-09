# to show output from the 'tests', run with 
# pytest skax_test.py  -rP

from functools import partial
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=3)
import scipy.stats
import einops
import matplotlib
from functools import partial
from collections import namedtuple
import jax
import jax.random as jr
import jax.numpy as jnp
from jax import vmap, grad, jit
import jax.scipy as jsp
import itertools
from itertools import repeat
from time import time
import chex
import typing

import jax
from typing import Any, Callable, Sequence
from jax import lax, random, numpy as jnp
from flax import linen as nn
import flax

import jaxopt
import optax

import sklearn.datasets
import sklearn
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from skax import *
#jax.config.update("jax_enable_x64", True) # jaxopt.lbfgs uses float32

def print_probs(probs):
    str = ['{:0.3f}'.format(p) for p in probs]
    print(str)

def make_data(seed, n_samples, class_sep, n_features):
    X, y = sklearn.datasets.make_classification(n_samples=n_samples, n_features=n_features,  n_informative=5,
    n_redundant=5, n_repeated=0, n_classes=10, n_clusters_per_class=1, weights=None, flip_y=0.01,
    class_sep=class_sep, hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=seed)
    return X, y


### Compare sklearn logreg pipeline with skax logreg pipeline  (both use LBFGS)

def fit_classifier_sklearn(key, X, Y):
    classifier = Pipeline([
            ('standardscaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2)), 
            ('logreg', LogisticRegression(random_state=0, max_iter=100, C=1e5))])
    classifier.fit(np.array(X), np.array(Y))
    return classifier

def fit_classifier_skax_mlp(key, X, Y):
    nclasses  = len(np.unique(Y))
    network = MLPNetwork((nclasses,))
    model = NeuralNetClassifier(network, key, nclasses, l2reg=1e-5, max_iter=500, optimizer='lbfgs')
    classifier = Pipeline([
            ('standardscaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2)), 
            ('logreg', model)])
    classifier.fit(np.array(X), np.array(Y))
    return classifier

def test_logreg_pipeline():    
    key = jr.PRNGKey(0)
    X, y = make_data(0, n_samples=500, class_sep=1, n_features=10)
    clf = fit_classifier_sklearn(key, X, y)
    true_probs = clf.predict_proba(X)
    model = fit_classifier_skax_mlp(key, X, y)
    probs = np.array(model.predict(X))
    delta = np.max(true_probs - probs)
    print('max deviation from true probs {:.3f}'.format(delta))
    print('truth'); print_probs(true_probs[0])
    print('pred'); print_probs(probs[0])
    assert delta < 0.1

#### Compare sklearn logreg with skax logreg using different stochastic optimizers

def fit_predict_logreg(Xtrain, ytrain, Xtest, ytest, l2reg):
    # Use sklearn to fit logistic regression model
    classifier = Pipeline([
            ('standardscaler', StandardScaler()), 
            ('logreg', LogisticRegression(random_state=0, max_iter=100, C=1/l2reg))])
    classifier.fit(Xtrain, ytrain)
    train_probs = classifier.predict_proba(Xtrain)
    test_probs = classifier.predict_proba(Xtest)
    return train_probs, test_probs

def compare_probs(logreg_probs, probs, labels):
    delta = np.max(logreg_probs - probs)
    logreg_pred = np.argmax(logreg_probs, axis=1)
    pred = np.argmax(probs, axis=1)
    n = len(labels)
    logreg_error_rate = np.sum(logreg_pred != labels)/n
    error_rate = np.sum(pred != labels)/n
    #print(f'max difference in probs from {name} to logreg = {delta:.3f}')
    #print('error rates: logreg={:.3f}, model={:.3f}'.format(logreg_error_rate, error_rate))
    #print_probs(logreg_probs[0])
    #print_probs(probs[0])
    return delta, logreg_error_rate, error_rate


def compare_logreg(optimizer, name=None, batch_size=None, num_epochs=50,
                   n_samples=500, class_sep=1, n_features=10):
    key = jr.PRNGKey(0)
    l2reg = 1e-5
    X_train, y_train = make_data(0, n_samples, class_sep, n_features)
    X_test, y_test = make_data(1, 1000, class_sep, n_features)
    nclasses = len(np.unique(y_train))
    train_probs_logreg, test_probs_logreg = fit_predict_logreg(X_train, y_train, X_test, y_test, l2reg)
    
    #network = MLPNetwork((5, nclasses,))
    network = MLPNetwork((nclasses,)) # no hidden layers == logistic regression
    model = NeuralNetClassifier(network, key, nclasses, l2reg=l2reg, optimizer = optimizer, 
            batch_size=batch_size, num_epochs=num_epochs, print_every=0)  
    model.fit(X_train, y_train)
    train_probs = np.array(model.predict(X_train))
    test_probs = np.array(model.predict(X_test))

    train_delta, train_logreg_error_rate, train_error_rate = compare_probs(train_probs_logreg, train_probs, y_train)
    test_delta, test_logreg_error_rate, test_error_rate = compare_probs(test_probs_logreg, test_probs, y_test)
    print('max difference in probabilities from logreg to {:s}={:.3f}'.format(name, train_delta))
    #print('misclassification rates: logreg train = {:.3f}, model train = {:.3f}'.format(
    #    train_logreg_error_rate, train_error_rate))
    #print('misclassification rates: logreg test = {:.3f}, model test = {:.3f}'.format(
    #    test_logreg_error_rate, test_error_rate))



def test_logreg():
    compare_logreg("lbfgs", name="lbfgs")
    compare_logreg("armijo", name="armijo, bs=32", batch_size=32)
    compare_logreg(optax.adam(1e-3), name="adam 1e-3, bs=32", batch_size=32)
    compare_logreg("adam+warmup", name="adam+warmup, bs=32", batch_size=32)

def test_adam_warmup_robustness():
    compare_logreg("adam+warmup", name="adam+warmup, bs=32, N=500, sep=0.5, D=10", batch_size=32,
        n_samples=500, class_sep=0.5, n_features=10)

    compare_logreg("adam+warmup", name="adam+warmup, bs=32, N=500, sep=0.5, D=50", batch_size=32,
        n_samples=500, class_sep=0.5, n_features=50)

    compare_logreg("adam+warmup", name="adam+warmup, bs=32, N=500, sep=0.5, D=100", batch_size=32,
        n_samples=500, class_sep=0.5, n_features=100)

    compare_logreg("adam+warmup", name="adam+warmup, bs=32, N=1000, sep=0.5, D=100", batch_size=32,
        n_samples=1000, class_sep=0.5, n_features=100)
    
    compare_logreg("adam+warmup", name="adam+warmup, bs=32, N=1000, sep=1, D=100", batch_size=32,
        n_samples=1000, class_sep=1, n_features=100)


### Create synthetic dataset from GMM, compare MLP predictions (with different optimizers) to Bayes optimal

@chex.dataclass
class GenParams:
    nclasses: int
    nfeatures: int
    prior: chex.Array
    mus: chex.Array # (C,D)
    Sigmas: chex.Array #(C,D,D)

def make_params(key, nclasses, nfeatures, scale_factor=1):
    mus = jr.normal(key, (nclasses, nfeatures)) # (C,D)
    # shared covariance -> linearly separable
    #Sigma = scale_factor * jnp.eye(nfeatures)
    #Sigmas = jnp.array([Sigma for _ in range(nclasses)]) # (C,D,D)
    # diagonal covariance -> nonlinear decision boundaries
    sigmas = jr.uniform(key, shape=(nclasses, nfeatures), minval=0.5, maxval=5)
    Sigmas = jnp.array([scale_factor*jnp.diag(sigmas[y]) for y in range(nclasses)])
    prior = jnp.ones(nclasses)/nclasses
    return GenParams(nclasses=nclasses, nfeatures=nfeatures, prior=prior, mus=mus, Sigmas=Sigmas)

def sample_data(key, params, nsamples):
    y = jr.categorical(key, logits=jnp.log(params.prior), shape=(nsamples,))
    X = jr.multivariate_normal(key, params.mus[y], params.Sigmas[y])
    return X, y

def predict_bayes(X, params):
    def lik_fn(y):
        return jsp.stats.multivariate_normal.pdf(X, params.mus[y], params.Sigmas[y])
    liks = vmap(lik_fn)(jnp.arange(params.nclasses)) # liks(k,n)=p(X(n,:) | y=k)
    joint = jnp.einsum('kn,k -> nk', liks, params.prior) # joint(n,k) = liks(k,n) * prior(k)
    norm = joint.sum(axis=1) # norm(n)  = sum_k joint(n,k) = p(X(n,:)
    post = joint / jnp.expand_dims(norm, axis=1) # post(n,k) = p(y = k | xn)
    return post

def compare_bayes(optimizer, name, nhidden, scale_factor):
    nclasses = 4
    key = jr.PRNGKey(0)
    key, subkey = jr.split(key)
    params = make_params(subkey, nclasses=nclasses, nfeatures=10, scale_factor=scale_factor)
    key, subkey = jr.split(key)
    Xtrain, ytrain = sample_data(subkey, params, nsamples=1000)
    key, subkey = jr.split(key)
    Xtest, ytest = sample_data(subkey, params, nsamples=1000)

    yprobs_train_bayes = predict_bayes(Xtrain, params)
    yprobs_test_bayes = predict_bayes(Xtest, params)

    ypred_train_bayes = jnp.argmax(yprobs_train_bayes, axis=1)
    error_rate_train_bayes = jnp.sum(ypred_train_bayes != ytrain) / len(ytrain)

    ypred_test_bayes = jnp.argmax(yprobs_test_bayes, axis=1)
    error_rate_test_bayes = jnp.sum(ypred_test_bayes != ytest) / len(ytest)

    nhidden = nhidden + (nclasses,) # set nhidden() to get logistic regression
    network = MLPNetwork(nhidden)
    mlp = NeuralNetClassifier(network, key, nclasses, l2reg=1e-5, optimizer = optimizer, 
            batch_size=32, num_epochs=30, print_every=0)  
    mlp.fit(Xtrain, ytrain)

    yprobs_train_mlp = np.array(mlp.predict(Xtrain))
    yprobs_test_mlp = np.array(mlp.predict(Xtest))

    ypred_train_mlp = jnp.argmax(yprobs_train_mlp, axis=1)
    error_rate_train_mlp = jnp.sum(ypred_train_mlp != ytrain) / len(ytrain)

    ypred_test_mlp = jnp.argmax(yprobs_test_mlp, axis=1)
    error_rate_test_mlp = jnp.sum(ypred_test_mlp != ytest) / len(ytest)
    
    delta_train = jnp.max(yprobs_train_bayes - yprobs_train_mlp)
    delta_test = jnp.max(yprobs_test_bayes - yprobs_test_mlp)

    print('Evaluating training method {:s} on model with {} hidden layers'.format(name, nhidden))
    print('Train error rate {:.3f} (Bayes {:.3f}), Test error rate {:.3f} (Bayes {:.3f})'.format(
        error_rate_train_mlp, error_rate_train_bayes, error_rate_test_mlp, error_rate_test_bayes))
    #print('Max diff in probs from Bayes: train {:.3f}, test {:.3f}'.format(
    #    delta_train, delta_test))
    print('\n')

def test_mlp_vs_bayes_sf1():
    compare_bayes(optax.adam(1e-3), "adam(1e-3)", nhidden=(), scale_factor=1)
    compare_bayes("adam+warmup", "adam+warmup", nhidden=(), scale_factor=1)

    compare_bayes(optax.adam(1e-3), "adam(1e-3)", nhidden=(10,), scale_factor=1)
    compare_bayes("adam+warmup", "adam+warmup", nhidden=(10,), scale_factor=1)

    compare_bayes(optax.adam(1e-3), "adam(1e-3)", nhidden=(10,10), scale_factor=1)
    compare_bayes("adam+warmup", "adam+warmup", nhidden=(10,10), scale_factor=1)


def test_mlp_vs_bayes_sf5():
   # scale_factor = 5 means the class conditional densities have higher variance (more overlap)

    compare_bayes(optax.adam(1e-3), "adam(1e-3)", nhidden=(), scale_factor=5)
    compare_bayes("adam+warmup", "adam+warmup", nhidden=(), scale_factor=5)

    compare_bayes(optax.adam(1e-3), "adam(1e-3)", nhidden=(10,), scale_factor=5)
    compare_bayes("adam+warmup", "adam+warmup", nhidden=(10,), scale_factor=5)

    compare_bayes(optax.adam(1e-3), "adam(1e-3)", nhidden=(10,10), scale_factor=5)
    compare_bayes("adam+warmup", "adam+warmup", nhidden=(10,10), scale_factor=5)
