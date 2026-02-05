import numpy as np
from raw.c5.helper import assert_dim, assert_same_shape, _pad_1d
# forward 1d convolution..

def conv_1d(inp: np.ndarray, param: np.ndarray) -> np.ndarray:
  assert_dim(inp, 1)
  assert_dim(param, 1)
  # pad input based on param length
  # param_mid = param.shape[0] // 2
  inp_pad = _pad_1d(inp, param.shape[0] // 2)
  # initialize output.  it is the same shape as inp here because we are not downsampling
  out = np.zeros(inp.shape)

  # the actual convolution
  for o in range(len(out)):
    inp_win = inp_pad[o:o+len(param)]
    out[o] = np.dot(inp_win, param)

  assert_same_shape(inp, out)
  return out



