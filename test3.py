import matplotlib.pyplot as plt

data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6]

# Creating histograms with different bin values
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(data, bins=5, color='blue', edgecolor='black')
plt.title('Histogram with 5 Bins')

plt.subplot(1, 2, 2)
plt.hist(data, bins=10, color='green', edgecolor='black')
plt.title('Histogram with 10 Bins')

plt.tight_layout()
plt.show()