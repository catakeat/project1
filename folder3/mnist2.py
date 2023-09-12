from sklearn.datasets import fetch_openml
import matplotlib as plt

mnist = fetch_openml('mnist_784', version=1,parser='auto')
X,y = mnist["data"],mnist["target"]


print("In mnist2")
some_digit = X[0]
some_digit_image = some_digit.reshape(28,28)
print(some_digit_image)
