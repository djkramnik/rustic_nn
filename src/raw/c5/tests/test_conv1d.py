import numpy as np
from raw.c5.conv_1d import _conv_grads_1d, _input_grad_1d, conv_1d

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

  [inp_grad, param_grad] = _conv_grads_1d(input_1d, param_1d)
  assert np.allclose(inp_grad, np.array([2, 3, 3, 3, 2]))
  assert np.allclose(param_grad, np.array([10, 15, 14]))

def test_slow_brain():
  input_1d = np.array([1,2,3,4,5])
  param_1d = np.array([1,1,1])
  evil_out_grad = np.array([1, 2, 3, 4, 5])
  ans = _input_grad_1d(input_1d, param_1d, evil_out_grad)
  assert np.allclose(ans, np.array([3, 6, 9, 12, 9]))
  [ans2, param_grad] = _conv_grads_1d(input_1d, param_1d, evil_out_grad)
  assert np.allclose(ans, ans2)

def test_injured_brain():
  inp = np.array([1, 2, 3, 4])
  param = np.array([5, 6, 7])
  evil_out_grad = np.array([2, 3, 4, 5])
  ans = _input_grad_1d(inp, param)
  assert np.allclose(ans, [11, 18, 18, 13])
  ans2 = _input_grad_1d(inp, param, evil_out_grad)
  assert np.allclose(ans2, [27, 52, 70, 58])



