import numpy as np
from raw.activation import sigmoid, sigmoid_deriv, sigmoid_deriv_elegant

def test_sigmoid():
  a = np.array([2])
  expected = 1 / (1 + np.exp(-2))

  output_a = sigmoid(a)
  assert output_a.shape == a.shape
  assert np.allclose(output_a, expected)

  b = np.array([[2, 2], [2, 2]])
  output_b = sigmoid(b)
  assert output_b.shape == b.shape == (2,2)
  assert np.allclose(output_b, expected)

def test_sigmoid_deriv():
  a = np.array([0])
  assert(np.allclose(sigmoid_deriv(a), 0.25))
  assert(np.allclose(sigmoid_deriv_elegant(a), 0.25))