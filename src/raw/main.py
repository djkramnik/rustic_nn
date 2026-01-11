import numpy as np

from raw.helper import *
from raw.loss import softmax, softmax_safe
from data.load_mnist import load
import util.np_utils as utils

X_train, y_train, X_test, y_test = load()


y_test_encoded = one_hot_encode(y_test, 10)
y_train_encoded = one_hot_encode_fast(y_train, 10)

X_train, X_test = scale_input_data(X_train, X_test)

print(X_train.shape)

# initialize random weights..
layer_one_weights = np.random.randn(X_train.shape[1], 89)
layer_one_bias = np.zeros((1, 89))

layer_two_weights = np.random.randn(89, 10)
layer_two_bias = np.zeros((1, 10))

def forward(input: np.ndarray, params: np.ndarray):
  return np.dot(input, params)

def addbias(input: np.ndarray, bias: np.ndarray):
  return input + bias

# need layer 1 weights

# before we proceed with the loop we need...
# sigmoid and sigmoid deriv function

# impl me please

# need a GENERATOR.  not just one batch, but a generator for a set of batches
# that covers the entire training set and comprises an EPOCH
def get_batch_generator(x: np.ndarray, y: np.ndarray, size:int = 60) -> np.ndarray:
  # get a slice of size from a shuffled x and the corresponding target
  return x, y

# impl me please
def sigmoid():
  pass

def linear():
  pass

def linear_deriv():
  pass

# impl me please
def sigmoid_deriv():
  pass

# we need a loop guy
MAX_EPOCHS = 50
epoch_i = 0
while(epoch_i < MAX_EPOCHS):
  # we need to get a random batch of the X_train data

  batch_gen = get_batch_generator(X_train, y_train_encoded)
  for ii, (xb, yb) in enumerate(batch_gen):
    layer_one_logits = forward(xb, layer_one_weights) + layer_one_bias
    layer_one_output = sigmoid(layer_one_logits)

    layer_two_logits = forward(layer_one_output, layer_two_weights) + layer_two_bias
    layer_two_output = linear(layer_two_logits)
    # calculate the loss
    # do backward
    # update (mutate) the weights

  # end of epoch.  recalculate the loss, including on test set?
  # and evaluate whether to continue
  epoch_i += 1



