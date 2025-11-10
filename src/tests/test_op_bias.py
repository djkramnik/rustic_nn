import numpy as np
from operation.bias_add import BiasAdd
from util.assert_shape import assert_shape

def test_bias_forward_and_backward():
    # Simple deterministic input and weight
    x = np.array([
      [1,2,3,4],
      [5,6,7,8]
    ])      # shape (2, 4)
    w = np.array([[1,1,1,1]])    # shape (1, 4)

    op = BiasAdd(w)

    # ----- Forward -----
    y = op.forward(x)
    assert np.allclose(y, [[2,3,4,5],[6,7,8,9]])

    # ----- Backward -----
    grad_out = np.array([
        [2,3,4,5],
        [6,7,8,9]
        ])
    grad_in = op.backward(grad_out)

    # the input gradient is the outgrad itself
    assert np.allclose(grad_in, grad_out)
    assert_shape(x, op.input_grad)

    # expected param grad is just the output grad summed across all rows
    expected_grad_w = np.array([[8, 10, 12, 14]])
    assert np.allclose(op.param_grad, expected_grad_w)
    assert_shape(w, op.param_grad)
