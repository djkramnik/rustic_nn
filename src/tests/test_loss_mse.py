import numpy as np
from loss.mse import MeanSquaredError

def test_mse_forward_and_backward():
    # Simple test case
    preds = np.array([[1.0], [2.0], [3.0]])
    target = np.array([[1.0], [1.0], [2.0]])

    # Manual expected values
         # [[0],[1],[1]]
    expected_loss = 2/3
    expected_grad = np.array([[0],[2/3],[2/3]])

    mse = MeanSquaredError()

    # Forward test
    loss = mse.forward(preds, target)
    assert np.isclose(loss, expected_loss), f"loss {loss} != expected {expected_loss}"

    # Backward test
    grad = mse.backward()
    assert grad.shape == preds.shape
    assert np.allclose(grad, expected_grad)
