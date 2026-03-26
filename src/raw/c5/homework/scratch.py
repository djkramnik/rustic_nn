import numpy as np

from raw.c5.helper import assert_dim
from raw.c5.homework.conv2d import conv_2d_param_grad
from raw.c5.homework.conv2d_full import conv_2d_full

a = np.array([1, 2, 3, 4]).reshape((2,2))

b = np.array([1, 2, 3, 2, 3, 1, 3, 1, 2]).reshape((3, 3))

def pad(arr, size = 1):
  padded_originals = [np.concatenate([np.zeros(size), row, np.zeros(size)]) for row in arr]
  rows = np.zeros((size, arr.shape[1] + (size * 2)))
  return np.concatenate([rows, padded_originals, rows], axis=0)

def conv_squares(arr, param):
  pw = param.shape[0]
  pad_sz = pw // 2
  padded_inp = pad(arr, pad_sz)
  output = np.zeros_like(arr)
  for i in range(arr.shape[0]):
    for j in range(arr.shape[1]):
      window = padded_inp[i:i+pw,j:j+pw]
      output[i][j] = np.sum(window * param)
  return output

# print(conv_squares(a, b))

param_grad = conv_2d_param_grad(a, b)
# print(param_grad)

outgrad = np.array([4,1,2,3]).reshape((2,2))
param_grad_2 = conv_2d_param_grad(a, b, outgrad)
# print(param_grad_2)


full_inp = np.array([1, 2, 3, 4, 2, 4, 1, 3]).reshape((1,2,2,2))
full_param = np.array([
  [
    [
      [1, 2, 3],
      [2, 3, 1],
      [3, 1, 2]
    ],
    [
      [1, 3, 1],
      [2, 2, 1],
      [3, 1, 1]
    ],
    [
      [2, 2, 2],
      [1, 1, 1],
      [3, 1, 1]
    ],
  ],
  [
    [
      [1, 1, 2],
      [3, 3, 3],
      [1, 2, 3]
    ],
    [
      [1, 2, 3],
      [3, 2, 1],
      [1, 2, 3]
    ],
    [
      [1, 1, 3],
      [3, 2, 1],
      [1, 2, 2]
    ],
  ],
])

def da_full_fwd(inp, param):
  assert_dim(inp, 4)
  assert_dim(param, 4)
  assert inp.shape[1] == param.shape[0]
  out = np.zeros((inp.shape[0], param.shape[1], inp.shape[2], inp.shape[3]))
  for obs in range(inp.shape[0]):
    for outchan in range(param.shape[1]):
      for inchan in range(param.shape[0]):
        out[obs][outchan] += conv_squares(inp[obs][inchan], param[inchan][outchan])
  return out

full_out = da_full_fwd(full_inp, full_param)
print(full_out.shape)

full_out_prior = conv_2d_full(full_inp, full_param)

assert np.allclose(full_out, full_out_prior)
