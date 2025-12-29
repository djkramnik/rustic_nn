import numpy as np

from raw.helper import *

from data.load_mnist import load

X_train, y_train, X_test, y_test = load()

print('raw nn coding')

y_test_encoded = one_hot_encode(y_test, 10)
print(y_test[0], y_test_encoded[0])

y_train_encoded = one_hot_encode_fast(y_train, 10)
print(y_train[0], y_train_encoded[0])