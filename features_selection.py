__author__ = 'giulio'

from sklearn.cross_validation import KFold, cross_val_score
import itertools
import numpy as np
import sys
from sklearn.utils import shuffle

def findsubsets(S, m):
    return list(itertools.combinations(S, m))

def exaustive_selection(clf, X, y, scoring='accuracy', fold=None):
    n_samples = X.shape[0]
    n_features = X.shape[1]
    if fold is None:
        fold = KFold(n_samples, n_folds=5)

    all_subsets = []
    s_set = set(np.arange(n_features).astype(int))
    for i in range(14, n_features+1):
        for s in findsubsets(s_set, i):
            all_subsets.append(s)

    n_comb = len(all_subsets)

    results = np.zeros(n_comb)

    sys.stdout.write("There we go...")
    for i, comb in enumerate(all_subsets):
        sys.stdout.write("\r%s/%s\n" % (i+1, n_comb))
        sys.stdout.flush()
        X_comb = X[:, comb]
        temp = np.zeros(5)
        for j in range(5):
            X_comb, y = shuffle(X_comb, y)
            temp[j] = cross_val_score(clf, X_comb, y, scoring=scoring, cv=fold).mean()
        results[i] = temp.mean()

    best_res = results.max()
    best_ind = results.argmax()
    best_comb = all_subsets[best_ind]
    return best_comb, best_res










