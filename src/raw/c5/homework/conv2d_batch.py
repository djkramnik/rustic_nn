import numpy as np
from numpy import ndarray

from raw.c5.helper import assert_dim
from raw.c5.homework.conv2d import conv_2d_fwd, pad_2d

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


