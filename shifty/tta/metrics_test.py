import pytest

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import sklearn.metrics  #import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

from shifty.tta.metrics  import *

def test_roc():
    # from https://towardsdatascience.com/roc-curve-and-auc-from-scratch-in-numpy-visualized-2612bb9459ab
    X, y = make_classification(n_samples=100, n_informative=10, n_features=20, flip_y=0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    prob_vector = model.predict_proba(X_test)[:, 1]

    fpr_sklearn, tpr_sklearn, thresholds_sklearn = sklearn.metrics.roc_curve(y_test, prob_vector)
    auc_sklearn = sklearn.metrics.roc_auc_score(y_test, prob_vector)
    fpr_kpm, tpr_kpm, thresholds_kpm = roc(y_test, prob_vector)
    auc_kpm = roc_auc(y_test, prob_vector)

    assert np.allclose(auc_kpm, auc_sklearn, atol=1e-2)

    plt.figure()
    plt.scatter(fpr_sklearn, tpr_sklearn, s=100, alpha=0.5, color="blue", label="Scikit-learn")
    plt.scatter(fpr_kpm, tpr_kpm, color="red", s=100, alpha=0.3, label="Our implementation")
    plt.title("ROC Curve: AUC sklearn {:.3f}, us {:.3f}".format(auc_sklearn, auc_kpm))
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()

@jit
def f(y_true, y_score):
    return roc_auc(y_true, y_score)

def test_jittable():
    y_true = jnp.array([0,    1,    0,   1])
    y_score = jnp.array([0.5, 0.6, 0.7, 0.8])
    auc_score = f(y_true, y_score)
    auc_sklearn = sklearn.metrics.roc_auc_score(y_true, y_score)
    assert np.allclose(auc_score, auc_sklearn, atol=1e-2)
    #fpr_sklearn, tpr_sklearn, thresholds_sklearn = sklearn.metrics.roc_curve(y_true, y_score)
    #fpr, tpr, thresholds = roc(y_true, y_score)