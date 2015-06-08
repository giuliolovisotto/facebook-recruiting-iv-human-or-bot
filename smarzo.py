__author__ = 'giulio'
import utils as ut
import features_selection as fs
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import scale

if __name__ == "__main__":

    X, y = ut.load_X(), ut.load_y()
    print X.shape, y
    X = scale(X)
    clf = GradientBoostingClassifier()
    X = X[:, :11]
    fs.exaustive_selection(clf, X, y, fold=StratifiedKFold(y, n_folds=5))

