import numpy as np
from data.load_mnist import load
from util.one_hot_encode import one_hot_encode
from util.scale import scale_train_data

X_train, y_train, X_test, y_test = load()

num_mnist_labels = 10

# one hot encode targets
y_train_encoded = one_hot_encode(y_train, num_mnist_labels)
y_test_encoded = one_hot_encode(y_test, num_mnist_labels)

# scale train data (mean = 0, variance = 1)
X_train, X_test = scale_train_data(X_train, X_test)


