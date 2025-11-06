import numpy as np
from operation.weightmultiply import WeightMultiply
from util.assert_shape import assert_shape

def test_weightmultiply_forward_and_backward():
    # Simple deterministic input and weight
    x = np.array([[1.0, 2.0]])      # shape (1, 2)
    w = np.array([[3.0], [4.0]])    # shape (2, 1)

    op = WeightMultiply(w)

    # ----- Forward -----
    y = op.forward(x)
    # y = x @ w = [[1*3 + 2*4]] = [[11]]
    assert np.allclose(y, [[11.0]])

    # ----- Backward -----
    grad_out = np.array([[1.0]])    # dL/dy = 1 for simplicity
    grad_in = op.backward(grad_out)

    # Expected gradients:
    # dL/dx = grad_out @ w.T = [[1*3, 1*4]] = [[3, 4]]
    expected_grad_in = np.array([[3.0, 4.0]])
    assert np.allclose(grad_in, expected_grad_in)
    assert_shape(x, op.input_grad)

    # dL/dw = x.T @ grad_out = [[1], [2]]
    expected_grad_w = np.array([[1.0], [2.0]])
    assert np.allclose(op.param_grad, expected_grad_w)
    assert_shape(w, op.param_grad)
