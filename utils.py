__author__ = 'giulio'
import numpy as np

scalef6 = 1000.0
scalef7 = 1000000.0

def load_X():
    X = np.loadtxt("data/features.csv")
    X[:, 6] = X[:, 6]/scalef6
    X[:, 7] = X[:, 7]/scalef7
    return X

def load_y():
    y = np.loadtxt("data/train_ids.csv", dtype='str', usecols=(2, )).astype(float).astype(int)
    return y

def load_X_test():
    X = np.loadtxt("data/features_test.csv")
    X[:, 6] = X[:, 6]/scalef6
    X[:, 7] = X[:, 7]/scalef7
    return X






