__author__ = 'giulio'
import numpy as np

def load_X():
    X = np.loadtxt("data/features.csv",)
    return X

def load_y():
    y = np.loadtxt("data/train_ids.csv", dtype='str', usecols=(2, )).astype(float).astype(int)
    return y

def load_X_test():
    X = np.loadtxt("data/features_test.csv",)
    return X






