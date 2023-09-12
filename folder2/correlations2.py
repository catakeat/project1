import downloader_and_functions as hfile
import seaborn as sns
import matplotlib.pyplot as plt

local_HOUSING_PATH = hfile.os.path.join("..", hfile.HOUSING_PATH)

# Load housing data
housing = hfile.load_housing_data(housing_path=local_HOUSING_PATH)

# Define attributes for correlation analysis
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]

# Calculate correlations for selected attributes
corr_matrix = housing[attributes].corr()

# Create a correlation matrix heatmap using Seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()
