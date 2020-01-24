from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pickle
from sklearn.externals import joblib

mnist = fetch_openml('mnist_784', version=1)
X, y = mnist['data'], mnist['target']
y = y.astype(np.int)

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)

saved_model = pickle.dumps(knn_clf)
joblib.dump(knn_clf, 'knn_clf.pkl')
