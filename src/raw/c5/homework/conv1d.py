import numpy as np

from raw.c5.helper import assert_dim

def pad_1d(arr, size=1):
  assert_dim(arr, 1)
  pad = np.zeros(size)
  return np.concatenate((pad, arr, pad))

def conv1d_fwd(arr, param, pad=1):
  inp_pad = pad_1d(arr, pad)
  out = np.zeros_like(arr)
  for i in range(arr.shape[0]):
    for j in range(param.shape[0]):
      out[i] += inp_pad[i + j] * param[j]
  return out

inp = np.array([1, 2, 3, 4])
param = np.array([5, 6, 7])

out = conv1d_fwd(inp, param)
print(out)
assert np.allclose([20, 38, 56, 39], out)


def clamp_01(n):
  return min(max(0, n), 1)

def input_grad_1d(arr, param, pad = 1, outgrad = None):
  if outgrad is None:
    outgrad = np.ones_like(arr)

  inp_grad_pad = np.zeros_like(pad_1d(arr, pad))

  for i in range(arr.shape[0]):
    for j in range(param.shape[0]):
      inp_grad_pad[i + j] += param[j] * outgrad[i]
  return inp_grad_pad[pad:pad * -1]

inp_grad = input_grad_1d(inp, param)
print(inp_grad)
print(input_grad_1d(inp, param, 1, np.array([2, 3, 4, 5])))

def param_grad_1d(arr, param, pad = 1, outgrad = None):
  if outgrad is None:
    outgrad = np.ones_like(param)
  inp_pad = pad_1d(arr, pad)

  param_grad = np.zeros_like(param)
  for i in range(arr.shape[0]):
    for j in range(param.shape[0]):
      param_grad[j] += (inp_pad[i + j] * outgrad[j])
  return param_grad


param_g = param_grad_1d(inp, param)
print('param', param_g)
param_g2 = param_grad_1d(inp, param, 1, np.array([1, 2, 3]))
print('param with outgrad', param_g2)