import numpy as np
from numpy.lib.stride_tricks import as_strided

def pad(arr, size = 1):
  padded_originals = [np.concatenate([np.zeros(size), row, np.zeros(size)]) for row in arr]
  rows = np.zeros((size, arr.shape[1] + (size * 2)))
  return np.concatenate([rows, padded_originals, rows], axis=0)


def sliding_window(inp: np.ndarray, window_shape: tuple[int, int]):
  [pw, ph] = inp.shape
  [kw, kh] = window_shape
  out_w = pw - kw + 1
  out_h = ph - kh + 1
  assert out_w > 0
  assert out_h > 0
  row_stride, col_stride = inp.strides
  new_strides = (row_stride, col_stride, row_stride, col_stride)
  new_shape = (out_w, out_h, kw, kh)
  return as_strided(inp, new_shape, new_strides)

a = np.array([1, 2, 3, 4]).reshape((2,2))
b = np.array([1, 2, 3, 2, 3, 1, 3, 1, 2]).reshape((3, 3))
padded_inp = pad(a, 1)

super_inp = sliding_window(padded_inp, (3,3))
print(super_inp)