from sklearn.model_selection import StratifiedShuffleSplit
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("../datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def load_housing_data(housing_path = HOUSING_PATH):
    csv_path = os.path.join(housing_path,"housing.csv")
    print(csv_path)
    return pd.read_csv(csv_path)

housing = load_housing_data( housing_path=HOUSING_PATH)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)


for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
    housing_cat = housing[["ocean_proximity"]]
#print(type(housing_cat))
housing_cat_endoded =  ordinal_encoder.fit_transform(housing_cat)
print(housing_cat_endoded)
#print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))