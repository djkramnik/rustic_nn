import numpy as np
from numpy import ndarray

from raw.c5.conv_2d_channels import _pad_conv_input
from raw.c5.helper import assert_dim
from raw.c5.homework.conv2d import conv_2d_fwd, conv_2d_inp_grad
from raw.c5.homework.conv2d_batch import pad_2d_batch

# this is still only padding the last two dims?
def pad_2d_full(arr: ndarray, size = 1):
  assert_dim(arr, 4)
  padded_chans = [pad_2d_batch(chan, size) for chan in arr]
  return np.stack(padded_chans)

# cifar_imgs = np.random.randn(10, 3, 32, 32)

# padded_cifar = pad_2d_full(cifar_imgs, 1)
# padded_cifar_book = _pad_conv_input(cifar_imgs, 1)
# assert np.allclose(padded_cifar.shape, [10, 3, 34, 34])
# assert np.allclose(padded_cifar.shape, padded_cifar_book.shape)


def conv_2d_full(arr: ndarray, param: ndarray):
  assert_dim(arr, 4)
  assert_dim(param, 4)
  assert arr.shape[1] == param.shape[0]

  output = np.zeros((arr.shape[0], param.shape[1], arr.shape[2], arr.shape[3]))
  for i in range(arr.shape[0]):
    for j in range(param.shape[0]):
      for k in range(param.shape[1]):
        output[i][k] += conv_2d_fwd(arr[i][j], param[j][k])

  return output

def full_inp_grad(arr: ndarray, param: ndarray, outgrad: ndarray = None):
  if outgrad is None:
    outgrad = np.ones((arr.shape[0], param.shape[1], arr.shape[2], arr.shape[3]))

  inp_grad = np.zeros_like(arr)
  for i in range(arr.shape[0]):
    for j in range(param.shape[0]):
      for k in range(param.shape[1]):
        inp_grad[i][j] += conv_2d_inp_grad(arr[i][j], param[j][k], outgrad[i][k])
  return inp_grad

