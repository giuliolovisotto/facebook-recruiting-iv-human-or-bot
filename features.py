__author__ = 'giulio'

import pandas as pd
import numpy as np
import sys
import os
from os import listdir
from os.path import isfile, join
import joblib as jl
import time

import argparse

# File header looks like this
# bid_id,bidder_id,auction,merchandise,device,time,country,ip,url

# TODO: average delay from last bid, variance of delay from last bid

def single_bidder(b_id, b_hash, filtered):
    n_device = np.unique(filtered[:, 4]).shape[0]
    n_ip = np.unique(filtered[:, 7]).shape[0]
    n_country = np.unique(filtered[:, 6]).shape[0]
    n_url = np.unique(filtered[:, 8]).shape[0]

    df = pd.DataFrame(filtered[:, [1, 2]], index=filtered[:, 0], columns=['bid_id', 'auction'])
    bids_count = df.groupby(['auction']).count()['bid_id'].as_matrix()
    avg_b, var_b = bids_count.mean(), bids_count.var()

    t = filtered[:, 5].astype(float)
    t_diff = t[1:] - t[:-1]
    avg_t, var_t = t_diff.mean(), t_diff.var()
    zero_t = t_diff.shape[0] - np.count_nonzero(t_diff)

    f_array = np.array([
        n_device,
        n_ip,
        n_country,
        n_url,
        avg_b,
        var_b,
        avg_t,
        avg_b,
        zero_t
    ])

    f_array[np.isnan(f_array)] = 0

    np.savetxt("data/features/%s.csv" % b_id, f_array)
    # print f_array
    print b_id



if __name__ == "__main__":
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
    start = time.time()
    print "loading dataset..."
    _f_bids = pd.read_csv("./data/bids.csv",)
    print "%s sec" % (time.time()-start)

    _bidder_ids = np.loadtxt("./data/bidder_ids.csv", delimiter=",", dtype='str')

    n_samples = _bidder_ids.shape[0]
    n_features = 4

    print "removing files from queue..."
    start = time.time()

    onlyfiles = [f for f in listdir("data/features/") if isfile(join("data/features/", f))]
    onlyfiles = filter(lambda x: x[-4:] == ".csv", onlyfiles)

    onlyfiles = map(lambda x: x[:-4], onlyfiles)

    _bidder_ids = _bidder_ids[~np.in1d(_bidder_ids[:, 0], onlyfiles)]
    print "%s sec" % (time.time()-start)

    start = time.time()

    m_f_bids = _f_bids.as_matrix()

    jl.Parallel(n_jobs=jl.cpu_count())(
        # jl.delayed(single_bidder)(row[0], _f_bids.query('bidder_id == "%s"' % row[1])) for row in
        jl.delayed(single_bidder)(row[0], row[1], m_f_bids[m_f_bids[:, 1] == row[1]]) for row in _bidder_ids
    )

    # for row in _bidder_ids:
    #     single_bidder(row[0], row[1], m_f_bids[m_f_bids[:, 1] == row[1]])

    # when the shit is over save everything into a single file

    features = np.zeros(shape=(n_samples, n_features))
    for i in range(n_samples):
        features[i, :] = np.loadtxt("data/features/%s.csv" % i,)

    np.savetxt("data/features.csv", features, fmt="%d")



