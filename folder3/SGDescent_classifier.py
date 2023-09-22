import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import cross_val_score

np.set_printoptions(threshold=np.inf, linewidth=np.inf)
mnist = fetch_openml('mnist_784', version=1,parser='auto')
X,y = mnist["data"],mnist["target"]

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
y_train_5 = (y_train == '5')

sgd_clf = SGDClassifier(random_state=42)
print(cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy"))
