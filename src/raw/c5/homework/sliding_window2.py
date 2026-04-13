import numpy as np
from numpy.lib.stride_tricks import as_strided

# pad a 3dim array
def pad(arr, size = 1):
  out = np.zeros((
    arr.shape[0],
    arr.shape[1] + (size * 2),
    arr.shape[2] + (size * 2)
    ))
  for i, chan in enumerate(arr):
    out[i][size:(-1 * size),size:(-1 * size)] = chan
  return out

# pad 3d tests

# a = np.array([1, 2, 3, 4, 2, 1, 3, 2]).reshape((2, 2, 2))
# print(a)
# padded = pad(a)
# print(padded)

# b = np.arange(1, 13).reshape((3,4))
# # print(b)
# c = np.stack([b[0][::-1], b[1][::-1], b[2][::-1]])
# inp2 = np.stack([b, c])
# assert np.allclose(inp2.shape, (2, 3, 4))
# print(inp2)

# padded_irregular = pad(inp2, 2)
# assert np.allclose(padded_irregular.shape, (2, 7, 8))

ndarray = np.ndarray
def conv_chan_squares(arr: ndarray, param: ndarray):
  assert len(arr.shape) == 3
  assert len(param.shape) == 4
  pad_size = param.shape[3] // 2
  padded_inp = pad(arr, pad_size)
  out = np.zeros((param.shape[1], arr.shape[1], arr.shape[2]))

  for out_chan in range(param.shape[1]):
    for in_chan in range(param.shape[0]):
      # conv 2d in here
      square = padded_inp[in_chan]
      window = param[in_chan][out_chan]
      k = window.shape[0]
      w = arr.shape[1]
      h = arr.shape[2]
      local_out = np.zeros((w, h))
      for i in range(w):
        for j in range(h):
          local_out[i][j] = np.sum(square[i:i+k,j:j+k] * window)
      out[out_chan] += local_out
  return out

def sliding_window_3d(inp: ndarray, window_shape: tuple[int, int]):
  chan, Hp, Hw = inp.shape
  kh, kw = window_shape
  out_h = Hp - kh + 1
  out_w = Hw - kw + 1
  assert out_w > 0
  assert out_h > 0
  chan_stride, row_stride, col_stride = inp.strides
  new_strides = (chan_stride, row_stride, col_stride, row_stride, col_stride)
  new_shape = (chan, out_h, out_w, kh, kw)
  return as_strided(inp, new_shape, new_strides)

inp = np.array([1, 2, 3, 4, 2, 1, 3, 2]).reshape((2, 2, 2))
param = np.array([
  1, 2, 3,
  2, 3, 1,
  3, 1, 2,

  1, 1, 1,
  2, 2, 2,
  3, 3, 3,

  2, 2, 2,
  1, 1, 1,
  3, 1, 1,

  1, 1, 1,
  1, 2, 1,
  1, 3, 2,

  2, 1, 1,
  3, 1, 1,
  1, 1, 1,

  2, 2, 2,
  1, 1, 1,
  3, 3, 3,
]).reshape((2, 3, 3, 3))

classic_out = conv_chan_squares(inp, param)
# print(classic_out)

# print(inp.shape)
padded_inp = pad(inp)

print(padded_inp.shape)
three_d_win = sliding_window_3d(padded_inp, (3,3))
print(three_d_win.shape)
print(three_d_win)
