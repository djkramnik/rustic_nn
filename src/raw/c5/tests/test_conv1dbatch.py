import numpy as np
from raw.c5.conv_1d_batch import _conv_grads_1d_batch, conv_1d_batch, input_grad_1d_batch, param_grad_1d_batch

def test_conv_1d_batch():
  input_1d_batch = np.array([[0,1,2,3,4,5,6],
                           [1,2,3,4,5,6,7]])
  param_1d = np.array([1,1,1])
  ans = np.array([
    [1., 3., 6., 9., 12., 15., 11.],
    [3., 6., 9., 12., 15., 18., 13.]
  ])
  assert np.allclose(conv_1d_batch(input_1d_batch, param_1d), ans)

def test_conv_1d_batch_grad():
  input_1d_batch = np.array([[0,1,2,3,4,5,6],
                          [1,2,3,4,5,6,7]])
  param_1d = np.array([1,1,1])
  [input_grad, param_grad] = _conv_grads_1d_batch(input_1d_batch, param_1d)
  param_grad_ans = np.array([36, 49, 48])
  input_grad_ans = np.array([
    [2, 3, 3, 3, 3, 3, 2],
    [2, 3, 3, 3, 3, 3, 2]
  ])

  assert np.allclose(param_grad_ans, param_grad)
  assert np.allclose(input_grad_ans, input_grad)
  assert np.allclose(input_grad_ans, input_grad_1d_batch(input_1d_batch, param_1d))

def test_book_impl():
  input_1d_batch = np.array([[0,1,2,3,4,5,6], [1,2,3,4,5,6,7]])
  param_1d = np.array([1,1,1])
  output_grad = input_1d_batch.copy() + 1

  [input_grad, param_grad] = _conv_grads_1d_batch(input_1d_batch, param_1d, output_grad)
  input_grad_book = input_grad_1d_batch(input_1d_batch, param_1d, output_grad)
  param_grad_book = param_grad_1d_batch(input_1d_batch, param_1d, output_grad)
  assert np.allclose(input_grad_book, input_grad)
  assert np.allclose(param_grad_book, param_grad)
