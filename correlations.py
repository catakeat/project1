from sklearn.model_selection import StratifiedShuffleSplit
import os
import tarfile
from six.moves import urllib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

import setari

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH= os.path.join("datasets","housing")
HOUSING_URL = DOWNLOAD_ROOT+"datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
        tgz_path = os.path.join(housing_path, "housing.tgz")
        urllib.request.urlretrieve(housing_url, tgz_path)
        housing_tgz = tarfile.open(tgz_path)
        housing_tgz.extractall(path=housing_path)
        housing_tgz.close()
    else:#intra sus d evreme ce e gata facut folderul
         #print("housing path este" + housing_path)
         pass

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path,"housing.csv")
    return pd.read_csv(csv_path)


fetch_housing_data()
housing = load_housing_data()
#avem deja
#housing = housing.reset_index()
#print(housing.head())
housing['income_cat'] = np.ceil(housing['median_income']/1.5)
#print(type(housing))
#split = StratifiedShuffleSplit(n_splits=1,test_size=0.2, random_state=10)
#print(type(split))

'''
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
print(corr_matrix)
plt.show()
'''

# Select only numeric columns for correlation calculation
numeric_columns = housing.select_dtypes(include=[np.number])
print(numeric_columns.head())
print(numeric_columns.describe())
print("de aici")

'''
# Calculate the correlation matrix
corr_matrix = numeric_columns.corr()
print(corr_matrix.describe())
corr_matrix.plot(kind='scatter',x='longitude',y='latitude',alpha=0.4,s=housing["population"/100],label="population",
    figsize=(10, 7),
    c="median_house_value",
    cmap=plt.get_cmap("jet"),
    colorbar=True)
# Sort the correlations with respect to 'median_house_value'
sorted_correlations = corr_matrix["median_house_value"].sort_values(ascending=False)

# Print sorted correlations and the correlation matrix
print(sorted_correlations)
print(corr_matrix)
'''