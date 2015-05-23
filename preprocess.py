__author__ = 'giulio'

import pandas as pd
import numpy as np
m_f_bids = pd.read_csv("./data/bids.csv", usecols=['bidder_id', 'auction']).as_matrix()
_bidder_ids = np.loadtxt("./data/bidder_ids.csv", delimiter=",", dtype='str')
# bid_id,bidder_id,auction,merchandise,device,time,country,ip,url
for row in _bidder_ids:
    mask = m_f_bids[:, 0] == row[1]
    auctions = np.unique(m_f_bids[mask][:, 1])
    np.savetxt("data/auctions/%s.csv" % row[0], auctions, fmt="%s")
    print row[0]