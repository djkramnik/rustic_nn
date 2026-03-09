import numpy as np

from raw.c5.helper import assert_dim, assert_same_shape
from raw.c5.homework.conv1d import pad_1d
type ndarray = np.ndarray

def pad_2d(arr: ndarray, size=1):
  list = [pad_1d(obs, size) for obs in arr]
  row_pad = np.zeros((size, len(list[0])))
  return np.concatenate([row_pad, list, row_pad])

def conv_2d_fwd(arr: ndarray, param: ndarray):
  pad = param.shape[0] // 2
  pad_arr = pad_2d(arr, pad)
  out = np.zeros_like(arr)
  plen = param.shape[0]
  # for loops here

  # iter across pad rows
  for i in range(arr.shape[0]):
    # iter across pad cols
    for j in range(arr.shape[1]):
      for p in range(plen):
        out[i][j] += np.sum(param[p] * pad_arr[i + p][j:j+plen])
  return out

# inp backprop (sans, incl. outgrad)

# arr is a 2d square tensor, as is param
def conv_2d_inp_grad(arr: ndarray, param: ndarray, outgrad: ndarray = None):
  if outgrad is None:
    outgrad = np.ones_like(arr)
  else:
    assert_same_shape(arr, outgrad)

  pad = param.shape[0] // 2
  plen = param.shape[0]
  inp_grad_padded = pad_2d(np.zeros_like(arr), pad)


  # each output is made up of different cells of the input, even though shape wise they are the same
  # when you are at the innermost loop running the update, remember that at the same location in fwd, you are updating a single output
  # position here.  That output gradient position is the one to multiply by
  for row in range(arr.shape[0]):
    for col in range(arr.shape[1]):
      for pr in range(plen):
        for pc in range(plen):
          inp_grad_padded[row + pr, col+pc] += (param[pr, pc] * outgrad[row , col])

  return inp_grad_padded[pad:-1,pad:-1]

# param backprop (sans, incl. outgrad)

def conv_2d_param_grad(arr: ndarray, param: ndarray, outgrad: ndarray = None):
  pass