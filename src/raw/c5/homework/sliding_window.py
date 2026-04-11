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
padded_inp = pad(a)

super_inp = sliding_window(padded_inp, (3,3))
# print(super_inp)

# a2 = np.arange(1, 13).reshape((3,4))
# b2 = np.array([1, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 2, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 2]).reshape((5,5))
# padded_inp2 = pad(a2, 2)
# super_inp2 = sliding_window(padded_inp2, b2.shape)
# print(super_inp2, super_inp2.shape)

# DOES NOT WORK
# test = np.matmul(super_inp, b)
# print(test)
# test2 = np.sum(test, axis=2)
# print(test2)
# print(super_inp.shape)
# print(super_inp)

# out = np.zeros((2,2))
# gah = None
# for i, arr in enumerate(super_inp[:]):
#   m = arr * b
#   # print(m.shape)
#   if i == 0:
#     gah = m

# print(gah)
# print(np.sum(gah, axis=(1,2)))

gah = super_inp * b
# print(gah.shape)
out = (super_inp * b)
ans1 = out.sum(axis=(2,3))
ans2 = out.sum(axis=(3,2))
ans3 = out.sum(axis=(-1,-2))

assert np.allclose(ans1, ans2, ans3)
print(ans1,'\n\n', ans2, '\n\n', ans3)