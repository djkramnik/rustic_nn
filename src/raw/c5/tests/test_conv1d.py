import numpy as np
from raw.c5.conv_1d import conv_1d

def test_conv_1d():
  input_1d = np.array([1,2,3,4,5])
  param_1d = np.array([1,1,1])
  result = np.sum(conv_1d(input_1d, param_1d))
  print(result)
  assert result == 39.

def test_conv_1d_grad():
  input_1d = np.array([1,2,3,4,5])
  param_1d = np.array([1,1,1])
  param_1d_2 = np.array([2,1,1])
  result1 = np.sum(conv_1d(input_1d, param_1d))
  result2 = np.sum(conv_1d(input_1d, param_1d_2))
  assert result2 - result1 == 10

  input_1d2 = np.array([2,3,4,5,6])
  result1 = np.sum(conv_1d(input_1d2, param_1d))
  result2 = np.sum(conv_1d(input_1d2, param_1d_2))
  assert result2 - result1 == 14
