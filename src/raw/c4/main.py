from typing import Iterator
import numpy as np

from raw.c4.activation import linear, sigmoid
from raw.c4.helper import *
from raw.c4.loss import binary_cross_entropy, softmax, softmax_ce, softmax_safe
from data.load_mnist import load
import util.np_utils as utils

X_train, y_train, X_test, y_test = load()


y_test_encoded = one_hot_encode(y_test, 10)
y_train_encoded = one_hot_encode_fast(y_train, 10)

X_train, X_test = scale_input_data(X_train, X_test)

print(X_train.shape)

# initialize random weights..
layer_one_weights = np.random.randn(X_train.shape[1], 89)
layer_one_bias = np.zeros((1, 89)) # (1, neurons)

layer_two_weights = np.random.randn(89, 10)
layer_two_bias = np.zeros((1, 10)) # (1, neurons)

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
def get_batch_generator(x: np.ndarray, y: np.ndarray, size:int = 60) -> Iterator[tuple[np.ndarray, np.ndarray]]:
  assert x.shape[0] == y.shape[0]
  perm = np.random.permutation(x.shape[0])
  for start in range(0, len(perm), size):
    chunk = perm[start:start+size]
    yield x[chunk], y[chunk]

# we need a loop guy
MAX_EPOCHS = 2000
epoch_i = 0


def loss_deriv(softmax_preds, target) -> np.ndarray:
  return (softmax_preds - target) / target.shape[0]

# this relies on the mutation of the weights so be careful
def calc_test_loss():
  N1 = forward(X_test, layer_one_weights)
  Z1 = N1 + layer_one_bias

  X2 = sigmoid(Z1)


  N2 = forward(X2, layer_two_weights)
  Z2 = N2  + layer_two_bias
  P = linear(Z2)
  # calculate the loss
  return softmax_ce(P, y_test_encoded)


best_loss = None
cached_layer_one_weights = layer_one_weights.copy()
cached_layer_one_bias = layer_one_bias.copy()
cached_layer_two_weights = layer_two_weights.copy()
cached_layer_two_bias = layer_two_bias.copy()

# calcuate from cached layer
# calculate preds without softmax..
# get the max pred value and compare it to the y_test value
def calc_accuracy_model(x_test, y_test):
  n1 = np.dot(x_test, cached_layer_one_weights)
  z1 = n1 + cached_layer_one_bias
  x2 = sigmoid(z1)
  n2 = np.dot(x2, cached_layer_two_weights)
  preds = n2 + cached_layer_two_bias
  acc = np.sum((np.argmax(preds, axis=1) == y_test).astype(int)) / y_test.shape[0]
  return acc

while(epoch_i < MAX_EPOCHS):
  # we need to get a random batch of the X_train and y_train data

  batch_gen = get_batch_generator(X_train, y_train_encoded)
  for ii, (xb, yb) in enumerate(batch_gen):

    N1 = forward(xb, layer_one_weights)
    Z1 = N1 + layer_one_bias

    X2 = sigmoid(Z1)


    N2 = forward(X2, layer_two_weights)
    Z2 = N2  + layer_two_bias

    P = linear(Z2)
    # calculate the loss
    preds = softmax_safe(P)

    # do backward

    #components
    #dense layer 1
    # X = (n,784)
    # W1 = (784, 89)
    # N1 = (n,89)
    # B1 = (1, 89)
    # Z1 (logits pre activation) = (n,89)

    #dense layer 2
    # X2 = sig(Z1) = (n,89)
    # W2 = (89,10)
    # N2 = (n, 10)
    # B2 = (1,10)
    # Z2 aka preds pre softmax = (n, 10)

    dN1dW1 = np.transpose(xb, (1, 0))
    # redundant
    dN1dX = np.transpose(layer_one_weights, (1, 0))

    dZ1dB1 = np.ones_like(layer_one_bias)
    dZ1dN1 = np.ones_like(N1)

    dX2dZ1 = X2 * (1 - X2)

    dN2dW2 = np.transpose(X2, (1,0))
    dN2dX2 = np.transpose(layer_two_weights, (1,0))

    dZ2dB2 = np.ones_like(layer_two_bias)
    dZ2dN2 = np.ones_like(N2)

    dLdZ2 = loss_deriv(preds, yb)

    dLdB2 = np.sum(dLdZ2 * dZ2dB2, axis=0, keepdims=True)

    dLdN2 = dLdZ2 * dZ2dN2

    dLdW2 = np.dot(dN2dW2, dLdN2)

    dLdX2 = np.dot(dLdN2, dN2dX2)

    dLdZ1 = dLdX2 * dX2dZ1

    dLdB1 = np.sum(dLdZ1 * dZ1dB1, axis=0, keepdims=True)
    dLdN1 = dLdZ1 * dZ1dN1

    dLdW1 = np.dot(dN1dW1, dLdN1)
    # redundant

    dLdX = np.dot(dLdN1, dN1dX)

    # check shapes
    # print('X', xb.shape)
    # print('dLdX', dLdX.shape)
    # print('W1', layer_one_weights.shape)
    # print('dLdW1', dLdW1.shape)
    # print('B1', layer_one_bias.shape)
    # print('dLdB1', dLdB1.shape)
    # print('X2', X2.shape)
    # print('dLdX2', dLdX2.shape)
    # print('W2', layer_two_weights.shape)
    # print('dLdW2', dLdW2.shape)
    # print('B2', layer_two_bias.shape)
    # print('dLdB2', dLdB2.shape)

    layer_one_weights -= (dLdW1 * 0.005)
    layer_one_bias -= (dLdB1 * 0.005)
    layer_two_weights -= (dLdW2 * 0.005)
    layer_two_bias -= (dLdB2 * 0.005)

  # end of epoch.  recalculate the loss on the entire test set bro
  # and evaluate whether to continue

  # calculate the loss after each epoch on the entire train set
  # cache the weight values
  # if the loss increases after an epoch than use the

  if (epoch_i > 0 and epoch_i % 10 == 0):
    new_loss = calc_test_loss()
    if (best_loss is None or new_loss < best_loss):
      print(f'loss on epoch {epoch_i}: {new_loss}')
      cached_layer_one_weights = layer_one_weights.copy()
      cached_layer_one_bias = layer_one_bias.copy()
      cached_layer_two_weights = layer_two_weights.copy()
      cached_layer_two_bias = layer_two_bias.copy()
      best_loss = new_loss
    else:
      print(f'regretably, on epoch {epoch_i + 1}, loss degraded from {best_loss} to {new_loss}.  abort.  abort it all')
      break

  epoch_i += 1

  # the best params (at the end of each epoch) are now safe in cached_ vars

print('FINAL ACCURACY: ', calc_accuracy_model(X_test, y_test))

