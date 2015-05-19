__author__ = 'giulio'
import numpy as np


def load_X():
    X = np.loadtxt("data/features.csv")
    X[:, 6] = X[:, 6]/1000.0
    X[:, 7] = X[:, 7]/1000000.0

    return X


def load_y():
    y = np.loadtxt("data/train_ids.csv", dtype='str', usecols=(2, )).astype(float).astype(int)
    return y




