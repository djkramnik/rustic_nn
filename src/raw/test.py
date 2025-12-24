import numpy as np

from data.load_mnist import load

X_train, y_train, X_test, y_test = load()

print('raw nn coding')
print(type(X_train))
print(X_train.shape)
print(y_test.shape)
