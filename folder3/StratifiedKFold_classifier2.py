import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection  import KFold
from sklearn.datasets  import load_digits

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

digits = load_digits()
print("shape",digits.data.shape)
print("target",len(digits.target))
kfolds = StratifiedKFold(n_splits=3,shuffle=True)
#kfolds = KFold(n_splits=3,shuffle=True)

for train_index,test_index in kfolds.split(digits.data,digits.target):
    #print("Train",train_index[:30])
    #print("Test",test_index[:30])
    X_train ,X_test,y_train,y_test = digits.data[train_index],digits.data[test_index],digits.target[train_index], digits.target[test_index]
