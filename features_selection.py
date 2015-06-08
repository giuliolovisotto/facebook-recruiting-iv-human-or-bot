__author__ = 'giulio'

from sklearn.cross_validation import KFold, cross_val_score
import itertools
import numpy as np
import sys
from sklearn.utils import shuffle

import joblib as jl

def single_eval(X, y, clf, fold, scoring, outp, i, length):
    temp = np.zeros(5)
    for j in range(5):
        X_comb, y = shuffle(X, y)
        temp[j] = cross_val_score(clf, X_comb, y, scoring=scoring, cv=fold, n_jobs=1).mean()
    print i*100/float(length)
    outp[i] = temp.mean()

def findsubsets(S, m):
    return list(itertools.combinations(S, m))

def exaustive_selection(clf, X, y, scoring='roc_auc', fold=None):
    n_samples = X.shape[0]
    n_features = X.shape[1]
    if fold is None:
        fold = KFold(n_samples, n_folds=5)

    all_subsets = []
    s_set = set(np.arange(n_features).astype(int))
    for i in range(4, n_features+1):
        for s in findsubsets(s_set, i):
            all_subsets.append(s)

    n_comb = len(all_subsets)

    results = np.memmap("tmp", dtype='float', shape=(n_comb, ), mode='w+')

    sys.stdout.write("There we go...")

    jl.Parallel(n_jobs=4)(
        jl.delayed(single_eval)(
            X[:, comb], y, clf, fold, scoring, results, i, len(all_subsets)
        ) for i, comb in enumerate(all_subsets)
    )

    results.flush()
    best_res = results.max()
    best_ind = results.argmax()
    best_comb = all_subsets[best_ind]
    print best_comb, best_res










