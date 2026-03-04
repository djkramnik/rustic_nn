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
  # for loops here