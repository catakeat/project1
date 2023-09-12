import downloader_and_functions as hfile
import pandas as pd
import numpy as np

local_HOUSING_PATH = hfile.os.path.join("..", hfile.HOUSING_PATH)

housing = hfile.load_housing_data( housing_path=local_HOUSING_PATH)
print(housing.head())
housing.drop('ocean_proximity',axis=1,inplace=True)
print(housing.head())
corr_matrix = housing.corr()
#hfile.plt.figure(figsize=(10, 8))
pd.plotting.scatter_matrix(corr_matrix, alpha=0.8, cmap='coolwarm', figsize=(12, 10))

hfile.plt.suptitle('Correlation Matrix Heatmap', fontsize=16)
hfile.plt.show()
