from data.load_mnist import load
from util.one_hot_encode import one_hot_encode

X_train, y_train, X_test, y_test = load()

num_mnist_labels = 10

y_train_encoded = one_hot_encode(y_train, num_mnist_labels)
y_test_encoded = one_hot_encode(y_test, num_mnist_labels)
