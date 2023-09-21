from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.linear_model import SGDClassifier


mnist = fetch_openml('mnist_784', version=1,parser='auto')

X,y = mnist["data"],mnist["target"]
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

sgd_clf = SGDClassifier(random_state=42)
y_train_5 = (y_train == '5') # True for all 5s, False for all other digits.
y_test_5 = (y_test == '5')
print(y_train_5)
''''
sgd_clf.fit(X_train, y_train_5)
some_digit = X.loc[1]
print(sgd_clf.predict([some_digit]))
some_digit_image = some_digit.values.reshape(28, 28)
plt.imshow(some_digit_image, cmap = plt.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()
'''

skfolds = StratifiedKFold(n_splits=3,random_state=42,shuffle=True)
#for train_index, test_index in skfolds.split(X_train,y_train)