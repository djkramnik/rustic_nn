import numpy as np
from numpy import ndarray

from raw.c5.helper import assert_dim, assert_same_shape
from raw.c5.homework.conv2d import conv_2d_fwd, conv_2d_inp_grad, conv_2d_param_grad, pad_2d

# this time we have batches of square tensors (batch_size, img_w, img_h)
def pad_2d_batch(arr: ndarray, size = 1):
  assert_dim(arr, 3)
  list = [pad_2d(square, size) for square in arr]
  return np.stack(list)

# arr is 3d, param is still 2d
def conv_2d_batch_fwd(arr: ndarray, param: ndarray):
  assert_dim(arr, 3)
  assert_dim(param, 2)
  list = [conv_2d_fwd(obs, param) for obs in arr]
  return np.stack(list)

def conv_2d_batch_inpgrad(arr: ndarray, param: ndarray, outgrad: ndarray = None):
  if outgrad is None:
    outgrad = np.ones_like(arr)
  else:
    assert_same_shape(arr, outgrad)

  grads = [conv_2d_inp_grad(obs, param, outgrad[i]) for (i, obs) in enumerate(arr)]
  return np.stack(grads)

def conv_2d_batch_paramgrad(
    arr: ndarray,
    param: ndarray,
    outgrad: ndarray=None):
  if outgrad is None:
    outgrad = np.ones_like(arr)
  else:
    assert_same_shape(arr, outgrad)

  param_grad = np.zeros_like(param)
  for (i, obs) in enumerate(arr):
    param_grad += conv_2d_param_grad(obs, param, outgrad[i])
  return param_grad


