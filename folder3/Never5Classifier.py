from sklearn.base import BaseEstimator
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1,parser='auto')
X,y = mnist["data"],mnist["target"]

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
y_train_5 = (y_train == '5')


class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)

result = cross_val_score(Never5Classifier(),X_train,y_train_5,cv=3,scoring="accuracy")
print(result)
