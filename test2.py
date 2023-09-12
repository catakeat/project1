import matplotlib.pyplot as plt

data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6]
print(len(data))
plt.hist(data,bins=10, color='blue', edgecolor='black')
print()
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Histogram example")
plt.show()