import numpy as np

from raw.c5.conv_2d_channels import _param_grad
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
  output = np.zeros(arr.shape)
  for i in range(arr.shape[0]):
    for j in range(arr.shape[1]):
      window = padded_inp[i:i+pw,j:j+pw]
      output[i][j] = np.sum(window * param)
  return output

def param_grad_squares(inp, param, outgrad = None):
  if (outgrad is None):
    outgrad = np.ones(inp.shape)
  else:
    assert np.allclose(outgrad.shape, inp.shape)
  pw = param.shape[0]
  pad_sz = pw // 2
  padded_inp = pad(inp, pad_sz)
  param_grad = np.zeros(param.shape)

  for i in range(inp.shape[0]):
    for j in range(inp.shape[1]):
      window = padded_inp[i:i+pw,j:j+pw]
      param_grad += (window * outgrad[i][j])
  return param_grad

# print(conv_squares(a, b))

ra = np.random.randn(2,2)
rb = np.random.randn(3,3)

param_grad = conv_2d_param_grad(ra, rb)
param_gradish = param_grad_squares(ra, rb)
assert np.allclose(param_grad, param_gradish)
# print(param_grad)

outgrad = np.random.randn(2,2)

# print(outgrad)
param_grad_2 = conv_2d_param_grad(ra, rb, outgrad)
param_gradish_2 = param_grad_squares(ra, rb, outgrad)
assert np.allclose(param_grad_2, param_gradish_2)


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
# print(full_out.shape)

full_out_prior = conv_2d_full(full_inp, full_param)

assert np.allclose(full_out, full_out_prior)

def da_full_paramgrad(inp, param, outgrad = None):
  assert_dim(inp, 4)
  assert_dim(param, 4)
  assert inp.shape[1] == param.shape[0]
  if outgrad is None:
    outgrad = np.ones((inp.shape[0], param.shape[1], inp.shape[2], inp.shape[3]))
  else:
    assert np.allclose(outgrad.shape, (inp.shape[0], param.shape[1], inp.shape[2], inp.shape[3]))

  param_grad = np.zeros(param.shape)

  for obs in range(inp.shape[0]):
    for outchan in range(param.shape[1]):
      for inchan in range(param.shape[0]):
        param_grad[inchan][outchan] += conv_2d_param_grad(inp[obs][inchan], param[inchan][outchan], outgrad[obs][outchan])

  return param_grad

fpg = da_full_paramgrad(full_inp, full_param)
neutral_full_outgrad = np.ones(full_out.shape)
bfpg = _param_grad(full_inp, neutral_full_outgrad, full_param)
assert np.allclose(fpg, bfpg)

evil_full_outgrad = np.random.randn(*full_out.shape)
fpg2 = da_full_paramgrad(full_inp, full_param, evil_full_outgrad)
bfpg2 = _param_grad(full_inp, evil_full_outgrad, full_param)
assert np.allclose(fpg2, bfpg2)