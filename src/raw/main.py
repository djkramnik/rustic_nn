import numpy as np

from raw.helper import *
from raw.loss import softmax, softmax_safe
from data.load_mnist import load
import util.np_utils as utils

X_train, y_train, X_test, y_test = load()

# print('raw nn coding')

y_test_encoded = one_hot_encode(y_test, 10)
# print(y_test[0], y_test_encoded[0])

y_train_encoded = one_hot_encode_fast(y_train, 10)
# print(y_train[0], y_train_encoded[0])

# now I need to scale the train data so that it has mean=0, variance = 1

# print('pre scaling input')
# print('mean', np.mean(X_train), np.mean(X_test))
# print('var', np.var(X_train), np.var(X_test))

X_train, X_test = scale_input_data(X_train, X_test)

# print('post scaling input')
# print('mean', np.mean(X_train), np.mean(X_test))
# print('var', np.var(X_train), np.var(X_test))

# for some reason testing home brewed std impl
# a = np.array([1, 2, 3, 100])
# print('home brew ok?', np.allclose(np.std(a), brew_std(a)))

print(X_train.shape)

# initialize random weights..
layer_one_weights = np.random.randn(X_train.shape[1], 89)
layer_one_bias = np.zeros((1, 89))

layer_two_weights = np.random.randn(89, 10)
layer_two_bias = np.zeros((1, 10))

#test homebrew softmax
# print('\nsoftmax\n')
# a = np.array([[1, 2, 5], [100, 100, 1]])
# brew2 = softmax_safe(a)

# summed = np.sum(brew2, axis=1)
# print('brew summed', summed)
# print('softmax_safe', brew2)

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
