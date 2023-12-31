from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.datasets  import load_digits
from sklearn.model_selection import train_test_split
from sklearn.model_selection  import KFold

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

digits = load_digits()
print("shape",digits.data.shape)
print("target",len(digits.target))
X_train,X_test,y_train,y_test=train_test_split(digits.data, digits.target, test_size=0.3)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train,y_train)
print(lr.score(X_test,y_test))

svm = SVC()
svm.fit(X_train,y_train)
print(svm.score(X_test,y_test))

rfc = RandomForestClassifier(n_estimators=40)
rfc.fit(X_train,y_train)
print(rfc.score(X_test,y_test))

kf=KFold(n_splits=3)
print(kf)  #KFold(n_splits=3, random_state=None, shuffle=False)

def clasifica(classifier):
    classifier.fit(X_train,y_train)
    print("Din clasifica",classifier.score(X_test,y_test))

clasifica(svm)




