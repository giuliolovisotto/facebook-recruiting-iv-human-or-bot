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

# TODO: contemporary active bids (need to find life time of b

def single_bidder(b_id, b_hash):
    print b_id
    filtered = np.loadtxt("data/f_bidder_id/%s.csv" % b_id, delimiter=',', dtype='str')
    auction_ids = np.loadtxt("data/auctions/%s.csv" % b_id, dtype='str', ndmin=1)
    n_auctions = 0
    if len(auction_ids) > 0:
        n_auctions = auction_ids.shape[0]

    if filtered.ndim == 1:
        if filtered.shape[0] == 0:
            np.savetxt("data/features/%s.csv" % b_id, np.zeros(17))
            f = open('data/emptyfiles.txt', 'a')
            f.write("%s\n" % b_id)
            f.close()
            return
        else:
            filtered = filtered[None, :]
    # if filtered.shape[0] == 0:
    # auction_id, merchandise, device, time, count, ip, url
    n_device = np.unique(filtered[:, 2]).shape[0]
    n_ip = np.unique(filtered[:, 5]).shape[0]
    n_country = np.unique(filtered[:, 4]).shape[0]
    n_url = np.unique(filtered[:, 6]).shape[0]
    n_categories = np.unique(filtered[:, 3].shape[0])

    df = pd.DataFrame(filtered[:, :2], columns=['auction', 'something'])
    bids_count = df.groupby(['auction']).count()['something']
    avg_b, std_b = bids_count.mean(), bids_count.std()

    df = pd.DataFrame(filtered[:, (2, 1)], columns=['device', 'something'])
    bids_count_device = df.groupby(['device']).count()['something']
    avg_b_d, std_b_d = bids_count_device.mean(), bids_count_device.std()

    df = pd.DataFrame(filtered[:, (6, 1)], columns=['url', 'something'])
    bids_count_url = df.groupby(['url']).count()['something']
    avg_b_url, std_b_url = bids_count_url.mean(), bids_count_url.std()

    df = pd.DataFrame(filtered[:, (5, 1)], columns=['ip', 'something'])
    bids_count_ip = df.groupby(['ip']).count()['something']
    avg_b_ip, std_b_ip = bids_count_ip.mean(), bids_count_ip.std()

    df = pd.DataFrame(filtered[:, (4, 1)], columns=['country', 'something'])
    bids_count_country = df.groupby(['country']).count()['something']
    avg_b_c, std_b_c = bids_count_country.mean(), bids_count_country.std()

    t = filtered[:, 3].astype(float)
    t_diff = t[1:] - t[:-1]
    avg_t, std_t = t_diff.mean(), t_diff.std()
    zero_t = t_diff.shape[0] - np.count_nonzero(t_diff)

    f_array = np.array([
        n_device,
        n_ip,
        n_country,
        n_url,
        n_categories,
        avg_b,
        std_b,
        avg_t/1000000.0,
        n_auctions,
        avg_b_d,
        std_b_d,
        avg_b_ip,
        std_b_ip,
        avg_b_url,
        std_b_url,
        avg_b_c,
        std_b_c
    ])

    f_array[np.isnan(f_array)] = 0

    np.savetxt("data/features/%s.csv" % b_id, f_array)
    # print f_array


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", action="store_true", help="remove preexisting files")
    parser.add_argument("-j", action="store_true", help="join feature files")
    parser.add_argument("--all", action="store_true", help="compute also test set features")

    args = parser.parse_args()

    _bidder_ids = np.loadtxt("./data/train_ids.csv", dtype='str', usecols=(0, 1))
    _test_ids = np.loadtxt("./data/test_ids.csv", dtype='str', usecols=(0, 1))

    n_train = _bidder_ids.shape[0]
    n_test = _test_ids.shape[0]

    if args.all:
        _bidder_ids = np.vstack((_bidder_ids, _test_ids))

    n_samples = _bidder_ids.shape[0]
    n_features = 17

    if args.j:
        print "joining files..."
        features = np.zeros(shape=(n_train, n_features))
        for i in range(n_train):
            features[i, :] = np.loadtxt("data/features/%s.csv" % i,)

        np.savetxt("data/features.csv", features)
        if args.all:
            features_test = np.zeros(shape=(n_test, n_features))
            for i in range(n_test):
                features_test[i, :] = np.loadtxt("data/features/%s.csv" % str(i+n_train),)

            np.savetxt("data/features_test.csv", features_test)
        sys.exit()

    if args.r:
        print "removing old files..."
        dirPath = "data/features/"
        fileList = os.listdir(dirPath)
        for fileName in fileList:
            os.remove(dirPath+"/"+fileName)
        if os.path.isfile("data/emptyfiles.txt"):
            os.remove("data/emptyfiles.txt")
        print "removed"

    print "removing files from queue..."
    start = time.time()

    onlyfiles = [f for f in listdir("data/features/") if isfile(join("data/features/", f))]
    onlyfiles = filter(lambda x: x[-4:] == ".csv", onlyfiles)

    onlyfiles = map(lambda x: x[:-4], onlyfiles)

    _bidder_ids = _bidder_ids[~np.in1d(_bidder_ids[:, 0], onlyfiles)]

    print "%s sec" % (time.time()-start)

    start = time.time()

    jl.Parallel(n_jobs=3)(
        # jl.delayed(single_bidder)(row[0], _f_bids.query('bidder_id == "%s"' % row[1])) for row in
        jl.delayed(single_bidder)(
            row[0],
            row[1],
        ) for row in _bidder_ids
    )

    # for row in _bidder_ids:
    #     single_bidder(row[0], row[1])




