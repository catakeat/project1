import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets  import load_digits

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

digits = load_digits()
print("shape",digits.data.shape)
print("target",len(digits.target))
X_train,X_test,y_train,y_test=train_test_split(digits.data, digits.target,test_size=0.3)
kfolds = StratifiedKFold(n_splits=3)

for train_index,test_index in kfolds.split(digits.data,digits.target):
    print("X_train",len(train_index))
    print("X_train", train_index)
    print("X_test", len(test_index))
    print("X_test",test_index)