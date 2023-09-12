from sklearn.impute import SimpleImputer
import os
import tarfile
from six.moves import urllib
import pandas as pd
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
# these are the unique ocean_proximity values

print(housing['ocean_proximity'].unique())

print(housing.head(100))
imputer = SimpleImputer(strategy="median")
# ocean proximty  is not a number , so get rid of it
housing_num = housing.drop("ocean_proximity",axis=1)
imputer.fit(housing_num)
#we see the medians for each column wheo evere has somethibg missing
#print(imputer.statistics_)
#print(housing_num.median())  ## show the median as a dataframe
#print(housing_num.median().values)# shows everything as an array exact imputer.statistics_
print("set 2")
print(housing_num.head(100))
