__author__ = 'giulio'
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
import utils as ut
import numpy as np
import os

reload(ut)

X_train, y_train = ut.load_X(), ut.load_y()

X_test = ut.load_X_test()
# mms = MinMaxScaler()
# X = mms.fit_transform(X)

rfc = ExtraTreesClassifier(bootstrap=False, class_weight=None, criterion='gini',
           max_depth=None, max_features='auto', max_leaf_nodes=None,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=1,
           oob_score=False, random_state=None, verbose=0, warm_start=False)
gbc = GradientBoostingClassifier(init=None, learning_rate=0.027825594022071243,
              loss='deviance', max_depth=3, max_features=None,
              max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=2,
              min_weight_fraction_leaf=0.0, n_estimators=100,
              random_state=None, subsample=1.0, verbose=0,
              warm_start=False)


rfc.fit(X_train, y_train)
gbc.fit(X_train, y_train)

rfc_y = rfc.predict_proba(X_test)[:, 1]
gbc_y = gbc.predict_proba(X_test)[:, 1]

alpha = 0.5

tot = alpha*rfc_y+(1-alpha)*gbc_y

_test_ids = np.loadtxt("./data/test_ids.csv", dtype='str', usecols=(1, ))

prediction = np.vstack((_test_ids, tot.astype(str))).T

hash_id = os.urandom(8).encode('hex')

np.savetxt(
    "data/subm/%s.csv" % hash_id, prediction, fmt="%s,%s", delimiter=",", header="bidder_id,prediction", comments=''
)



