import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets  import load_digits
from sklearn.linear_model import SGDClassifier
from sklearn.base import clone

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

sgd_clf = SGDClassifier(random_state=42)

digits = load_digits()
print("shape",digits.data.shape)
print("target",len(digits.target))
X_train,X_test,y_train,y_test = train_test_split(digits.data, digits.target,test_size=0.3)
y_train_5 = (y_train == 5) #
sgd_clf.fit(X_train,y_train_5)

skfolds = StratifiedKFold(n_splits=3)

for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct)
    print(n_correct / len(y_pred)) # prints 0.9502, 0.96565 and 0.96495