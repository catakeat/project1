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

kf=KFold(n_splits=3)
print(kf)  #KFold(n_splits=3, random_state=None, shuffle=False)
for train_index,test_index in kf.split([1,2,3,4,5,6,7,8,9]):
    print(train_index,test_index)

kf=KFold(n_splits=3,shuffle=True)
print(kf)  #KFold(n_splits=3, random_state=None, shuffle=False)
for train_index,test_index in kf.split([1,2,3,4,5,6,7,8,9]):
    print(train_index,test_index)





