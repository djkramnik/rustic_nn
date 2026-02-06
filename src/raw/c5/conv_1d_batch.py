import numpy as np
from raw.c5.helper import _pad_1d
from raw.c5.conv_1d import conv_1d

input_1d_batch = np.array([
  np.arange(0, 7),
  np.arange(1, 8)
])

# print(input_1d_batch)

# pad batch?  the slow way? ok..

def _pad_1d_batch(inp: np.ndarray, num: int) -> np.ndarray:
  outs = [_pad_1d(obs, num) for obs in inp]
  print(type(outs))
  return np.stack(outs)

# test = _pad_1d_batch(input_1d_batch, 1)
# print(test)

def conv_1d_batch(inp: np.ndarray, param: np.ndarray) -> np.ndarray:
  outs = [conv_1d(obs, param) for obs in inp]
  print(type(outs))
  return np.stack(outs)