from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pickle
from sklearn.datasets import fetch_openml
from sklearn.externals import joblib

knn_clf = joblib.load('knn_clf.pkl')
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist['data'], mnist['target']
y = y.astype(np.int)

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
knn_clf.predict(X_train)