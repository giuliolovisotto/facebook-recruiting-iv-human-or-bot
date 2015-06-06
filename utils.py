__author__ = 'giulio'
import numpy as np

def load_X():
    X = np.loadtxt("data/features.csv", usecols=(0,1,2,3,4,5,6,7,8,9,10))
    return X

def load_y():
    y = np.loadtxt("data/train_ids.csv", dtype='str', usecols=(2, )).astype(float).astype(int)
    return y

def load_X_test():
    X = np.loadtxt("data/features_test.csv", usecols=(0,1,2,3,4,5,6,7,8,9,10))
    return X






