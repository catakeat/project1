import downloader_and_functions as hfile

#fetch_housing_data()
housing = hfile.load_housing_data()
#print(housing.head())
#print(housing.info())
#print(housing.describe)
print(housing.count())

housing.hist(bins=50, figsize=(20,15))
hfile.plt.show()