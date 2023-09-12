import downloader_and_functions as hfile
import numpy as np

HOUSING_PATH = hfile.os.path.join("../",hfile.HOUSING_PATH)
#print(HOUSING_PATH)
housing = hfile.load_housing_data( housing_path=HOUSING_PATH)

def split_train_test(data,test_ratio):
    shuffled_indeces = np.random.permutation(len(data))
    test_set_size = int(len(data)*test_ratio)
    test_indices = shuffled_indeces[:test_set_size]
    train_indices = shuffled_indeces[test_set_size:]

    return data.iloc[train_indices],data.iloc[test_indices]

train_set, test_set = split_train_test(housing,0.2)
print(len(train_set),len(test_set))

from zlib import crc32
def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32
def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


housing_with_id = housing.reset_index() # adds an `index` column
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
s=housing["population"]/100, label="population", figsize=(10,7),
c="median_house_value", cmap=hfile.plt.get_cmap("jet"), colorbar=True,
)
hfile.plt.legend()
hfile.plt.show()
