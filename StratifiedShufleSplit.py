import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([0, 0, 0, 1, 1, 1])

sss= StratifiedShuffleSplit(n_splits=5,test_size=0.5,random_state=0)
for i,(train_index,test_index)  in enumerate(sss.split(X,y)):
    print(f"Fold {i}:")
    print(f"Train index {train_index}")
    print(f"Test index={test_index}")



