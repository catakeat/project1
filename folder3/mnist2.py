from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

mnist = fetch_openml('mnist_784', version=1,parser='auto')
X,y = mnist["data"],mnist["target"]
'''
print(X.shape)
print(y.shape)
print(y.info())
print(y.head())
'''
print("In mnist2")
print(X.info())
#jos e o coloana
#some_digit = X.iloc[0]

#jos e un rind
some_digit = X.loc[0]
#print(some_digit)
some_digit_image = some_digit.values.reshape(28,28)
print(some_digit_image)

plt.imshow(some_digit_image, cmap = "binary", interpolation="nearest")
plt.axis("off")
plt.show()
