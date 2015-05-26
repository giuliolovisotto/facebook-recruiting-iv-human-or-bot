__author__ = 'giulio'

from sklearn.cross_validation import KFold, cross_val_score
import itertools
import numpy as np
import sys

def findsubsets(S, m):
    return list(itertools.combinations(S, m))

def exaustive_selection(clf, X, y, scoring='accuracy', fold=None):
    n_samples = X.shape[0]
    n_features = X.shape[1]
    if fold is None:
        fold = KFold(n_samples, n_folds=5)

    all_subsets = []
    s_set = set(np.arange(n_features).astype(int))
    for i in range(1, n_features):
        for s in findsubsets(s_set, i):
            all_subsets.append(s)

    n_comb = len(all_subsets)

    results = np.zeros(n_comb)

    sys.stdout.write("There we go...")
    for i, comb in enumerate(all_subsets):
        sys.stdout.write("%s/%s" % (i+1, n_comb))
        X_comb = X[:, comb]
        results[i] = cross_val_score(clf, X_comb, y, scoring=scoring, cv=fold)

    best_res = results.max()
    best_ind = results.argmax()
    best_comb = all_subsets[best_ind]
    return best_comb









