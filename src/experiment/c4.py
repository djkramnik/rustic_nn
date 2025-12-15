from data.load_mnist import load

X_train, y_train, X_test, y_test = load()

print('testing', X_train.shape)