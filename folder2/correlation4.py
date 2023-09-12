import downloader_and_functions as hfile
import seaborn as sns
import matplotlib.pyplot as plt

local_HOUSING_PATH = hfile.os.path.join("..", hfile.HOUSING_PATH)

# Load housing data
housing = hfile.load_housing_data(housing_path=local_HOUSING_PATH)

housing.drop('ocean_proximity',axis=1,inplace=True)

housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]

hfile.pd.set_option('display.max_columns',None)
print(housing.head())
corr_matrix = housing.corr()
print(corr_matrix['median_house_value'].sort_values(ascending=False))