__author__ = 'giulio'
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
import utils as ut
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.preprocessing import scale, MinMaxScaler


reload(ut)

X_train, y_train = ut.load_X(), ut.load_y()



X_test = ut.load_X_test()

X = np.vstack((X_train, X_test))

mms = MinMaxScaler()
X = mms.fit_transform(X)

X_train = X[:X_train.shape[0]]
X_test = X[X_train.shape[0]:]

# mms = MinMaxScaler()
# X = mms.fit_transform(X)


clf1 = SVC(
    C=0.032442260791716297, cache_size=200, class_weight=None, coef0=0.0,
    degree=3, gamma=0.0, kernel='rbf', max_iter=-1, probability=True,
    random_state=None, shrinking=True, tol=0.001, verbose=False
)
clf2 = GradientBoostingClassifier(
    init=None, learning_rate=0.01, loss='deviance',
    max_depth=3, max_features=None, max_leaf_nodes=None,
    min_samples_leaf=1, min_samples_split=2, n_estimators=500,
    random_state=None, subsample=1.0, verbose=0,
)

clf1.fit(X_train, y_train)
clf2.fit(X_train, y_train)

clf1_y = clf1.predict_proba(X_test)[:, 1]
clf2_y = clf2.predict_proba(X_test)[:, 1]

alpha = 0.5

tot = alpha*clf1_y+(1-alpha)*clf2_y

_test_ids = np.loadtxt("./data/test_ids.csv", dtype='str', usecols=(1, ))

prediction = np.vstack((_test_ids, tot.astype(str))).T

hash_id = os.urandom(8).encode('hex')

np.savetxt(
    "data/subm/%s.csv" % hash_id, prediction, fmt="%s,%s", delimiter=",", header="bidder_id,prediction", comments=''
)



