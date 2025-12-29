import numpy as np

from raw.helper import *

from data.load_mnist import load

X_train, y_train, X_test, y_test = load()

print('raw nn coding')

y_test_encoded = one_hot_encode(y_test, 10)
print(y_test[0], y_test_encoded[0])

y_train_encoded = one_hot_encode_fast(y_train, 10)
print(y_train[0], y_train_encoded[0])

# now I need to scale the train data so that it has mean=0, variance = 1


# copy this
# model = NeuralNetwork(
#   layers=[
#     Dense(neurons=89, activation=Sigmoid()),
#     Dense(neurons=10, activation=Linear())
#   ],
#   loss = SoftmaxCrossEntropy(),
#   seed=sacred_seed
# )

def forward(input: np.ndarray, params: np.ndarray):
  return np.dot(input, params)

def addbias(input: np.ndarray, bias: np.ndarray):
  return input + bias

# need layer 1 weights
