__author__ = 'giulio'

import pandas as pd
import numpy as np
import sys
m_f_bids = pd.read_csv(
    "./data/bids.csv",
    usecols=['bidder_id', 'auction', 'merchandise', 'device', 'time', 'country', 'ip', 'url']
).as_matrix()
_train_ids = np.loadtxt("./data/train_ids.csv", dtype='str')[:, [0, 1]]
_test_ids = np.loadtxt("./data/test_ids.csv", dtype='str')
mapper = {row[1]: row[0] for row in np.vstack((_train_ids, _test_ids))}

_bidder_ids = np.vstack((_train_ids, _test_ids))
# bid_id,bidder_id,auction,merchandise,device,time,country,ip,url

all_auctions = np.unique(m_f_bids[:, 1])

sys.stdout.write("Step 1 of 2.\n")
for row in _bidder_ids:
    mask = (m_f_bids[:, 0] == row[1])
    auctions = np.unique(m_f_bids[mask][:, 1])
    # all_auctions = all_auctions.union(set(auctions))
    np.savetxt("data/f_bidder_id/%s.csv" % row[0], m_f_bids[mask][:, 1:], fmt="%s,%s,%s,%s,%s,%s,%s")
    np.savetxt("data/auctions/%s.csv" % row[0], auctions, fmt="%s")
    sys.stdout.write("\r%s/%s" % (int(row[0])+1, _bidder_ids.shape[0]))
    sys.stdout.flush()

n_a = len(all_auctions)

sys.stdout.write("Step 2 of 2.\n")

for i, a in enumerate(all_auctions):
    mask = (m_f_bids[:, 1] == a)
    masked = m_f_bids[mask]

    firstcol = masked[:, 0]
    firstcol = np.array([mapper[bidder] for bidder in firstcol])
    masked = np.hstack((firstcol[:, None], masked[:, [2, 4]]))

    np.savetxt("data/f_auctions/%s.csv" % a, masked, fmt="%s,%s,%s")
    sys.stdout.write("\r%s/%s" % (i+1, n_a))
    sys.stdout.flush()
