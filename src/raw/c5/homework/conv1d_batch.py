import numpy as np

from raw.c5.helper import assert_dim

from raw.c5.homework.conv1d import conv1d_fwd, input_grad_1d, pad_1d
from raw.c5.conv_1d_batch import _pad_1d_batch, conv_1d_batch, input_grad_1d_batch

def pad_1d_batch(arr, size=1):
  assert_dim(arr, 2)
  list = [pad_1d(obs, size) for obs in arr]
  return np.stack(list)

inp1 = np.array([1, 2, 3, 4])
inp2 = np.array([2, 3, 4, 5])
param = np.array([5, 6, 7])
batch = np.array([inp1, inp2])

padded = pad_1d_batch(batch)
book_padded = _pad_1d_batch(batch, 1)
assert np.allclose(padded, book_padded)

def conv1d_batch_fwd(arr, param, pad = 1):
  assert_dim(arr, 2)
  list = [conv1d_fwd(obs, param, pad) for obs in arr]
  return np.stack(list)

param = np.array([5, 6, 7])
out = conv1d_batch_fwd(batch, param)
out2 = conv_1d_batch(batch, param)
assert np.allclose(out, out2)

def inp_grad_batch(arr, param, pad = 1, outgrad = None):
  if outgrad is None:
    outgrad = np.ones_like(arr)

  list = [input_grad_1d(obs, param, pad, outgrad[i]) for (i, obs) in enumerate(arr)]
  return np.stack(list)

# inp_grad_1 = input_grad_1d(inp1, param)
# inp_grad_2 = input_grad_1d(inp2, param)
# print('inp_grad_1', inp_grad_1)
# print('inp_grad_2', inp_grad_2)

# batched_inp_grad = inp_grad_batch(batch, param)
# book_batched_inp_grad = input_grad_1d_batch(batch, param)
# assert np.allclose(batched_inp_grad, book_batched_inp_grad)
evil_out_grad = np.array([[1, 2, 3, 4], [1, 2, 3, 4]])

batched_inp_grad = inp_grad_batch(batch, param, 1, evil_out_grad)
book_batched_inp_grad = input_grad_1d_batch(batch, param, evil_out_grad)
assert np.allclose(batched_inp_grad, book_batched_inp_grad)
