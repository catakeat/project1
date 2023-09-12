from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1,parser='auto')
X,y = mnist['data'],mnist['target']
print(X.shape)
print(X.head(7))
print(y.shape)
#print(mnist.keys())
#print(X["pixel1"])