import numpy as np

from raw.c5.helper import assert_dim, assert_same_shape

from raw.c5.homework.conv1d import conv1d_fwd, input_grad_1d, pad_1d, param_grad_1d
from raw.c5.conv_1d_batch import _pad_1d_batch, conv_1d_batch, input_grad_1d_batch

def pad_1d_batch(arr, size=1):
  assert_dim(arr, 2)
  list = [pad_1d(obs, size) for obs in arr]
  return np.stack(list)

inp1 = np.array([1, 2, 3, 4])
inp2 = np.array([2, 3, 4, 5])
param = np.array([5, 6, 7])
batch = np.array([inp1, inp2])

padded = pad_1d_batch(batch)
book_padded = _pad_1d_batch(batch, 1)
assert np.allclose(padded, book_padded)

def conv1d_batch_fwd(arr, param):
  assert_dim(arr, 2)
  list = [conv1d_fwd(obs, param) for obs in arr]
  return np.stack(list)


def inp_grad_batch(arr, param, outgrad = None):
  assert_dim(arr, 2)
  assert_dim(param, 1)
  if outgrad is None:
    outgrad = np.ones_like(arr)
  else:
    assert_same_shape(arr, outgrad)

  list = [input_grad_1d(obs, param, outgrad[i]) for (i, obs) in enumerate(arr)]
  return np.stack(list)


def param_grad_batch(arr, param, outgrad=None):
  assert_dim(arr, 2)
  assert_dim(param, 1)
  if outgrad is None:
    outgrad=np.ones_like(arr)
  else:
    assert_same_shape(arr, outgrad)
  param_grad = np.zeros_like(param)
  for (i, obs) in enumerate(arr):
    param_grad += param_grad_1d(obs, param, outgrad[i])

  return param_grad