__author__ = 'giulio'

import pandas as pd
import numpy as np
import sys
import os
from os import listdir
from os.path import isfile, join

import time

import joblib as jl

import argparse


def single_bidder(b_id, b_hash, filtered):
    # filtered = (df.loc[df['bidder_id' == b_hash]])
    # filtered = df.query('bidder_id == "%s"' % b_hash)
    n_device = len(filtered['device'].unique())
    n_ip = len(filtered['ip'].unique())
    n_country = len(filtered['country'].unique())
    n_url = len(filtered['url'].unique())
    np.savetxt("data/features/%s.csv" % b_id, np.array([n_device, n_ip, n_country, n_url]), fmt="%d")
    print b_id


if __name__ == "__main__":
    _f_bids = pd.read_csv("./data/bids.csv")

    _bidder_ids = np.loadtxt("./data/bidder_ids.csv", delimiter=",", dtype='str')

    n_samples = _bidder_ids.shape[0]
    n_features = 4

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", action="store_true", help="remove preexisting files")
    args = parser.parse_args()

    if args.r:
        print "removing old files..."
        dirPath = "data/features/"
        fileList = os.listdir(dirPath)
        for fileName in fileList:
            os.remove(dirPath+"/"+fileName)
        print "removed"

    onlyfiles = [f for f in listdir("data/features/") if isfile(join("data/features/", f))]
    onlyfiles = filter(lambda x: x[-4:] == ".csv", onlyfiles)

    onlyfiles = map(lambda x: x[:-4], onlyfiles)

    for i, bidder in enumerate(_bidder_ids):
        if bidder[0] in onlyfiles:
            _bidder_ids[i, :] = np.array(['', ''])

    # print _bidder_ids[~np.any(_bidder_ids == '', axis=1)]

    _bidder_ids = _bidder_ids[~np.any(_bidder_ids == '', axis=1)]

    start = time.time()

    # jl.Parallel(jl.cpu_count())(
    #    jl.delayed(single_bidder)(row[0], row[1], _f_bids.query('bidder_id == "%s"' % row[1])) for row in
    #    _bidder_ids
    # )

    for row in _bidder_ids:
        single_bidder(row[0], row[1], _f_bids.query('bidder_id == "%s"' % row[1]))

    '''
    for i, bidder in enumerate(_bidder_ids):
        b_id = bidder[0]
        b_hash = bidder[1]
        filtered = _f_bids.query('bidder_id == "%s"' % b_hash)
        n_device = len(filtered['device'].unique())
        n_ip = len(filtered['ip'].unique())
        n_country = len(filtered['country'].unique())
        n_url = len(filtered['url'].unique())
        np.savetxt("data/features/%s.csv" % b_id, np.array([n_device, n_ip, n_country, n_url]), fmt="%d")
        print b_id
    '''

    # when the shit is over save everything into a single file

    features = np.zeros(shape=(n_samples, n_features))
    for i in range(n_samples):
        features[i, :] = np.loadtxt("data/features/%s.csv" % i,)

    np.savetxt("data/features.csv", features, fmt="%d")

    # np.savetxt("data/features.csv", features, fmt="%d")


