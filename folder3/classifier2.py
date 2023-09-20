from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.datasets  import load_digits


digits = load_digits()
print(digits.DESCR)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(digits.data.target,test_size=0.3)