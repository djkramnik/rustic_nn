# test_neural_network.py

import numpy as np
import neural_network.nn as nn_module

NeuralNetwork = nn_module.NeuralNetwork

def test_neural_network_train_batch():
    """
    Verify that NeuralNetwork:
    - calls layer.forward in order: layer0 -> layer1 -> ... -> layerN
    - calls layer.backward in reverse: layerN -> ... -> layer1 -> layer0
    - calls loss.forward once after the forward pass
    - calls loss.backward once before the backward pass
    """

    calls = []  # we'll record tuples like ("forward", "L0"), ("loss_forward",), etc.

    class SpyLayer:
        def __init__(self, name: str):
            # name is just for identification in the 'calls' list
            self.name = name
            self.last_forward_input = None
            self.last_backward_grad = None

            # mimic the API expected by NeuralNetwork.params()/param_grads()
            self.params = []
            self.param_grads = []

        def forward(self, x: np.ndarray) -> np.ndarray:
            calls.append(("forward", self.name))
            self.last_forward_input = x
            # identity: output == input
            return x

        def backward(self, grad: np.ndarray) -> np.ndarray:
            calls.append(("backward", self.name))
            self.last_backward_grad = grad
            # identity gradient: pass grad through unchanged
            return grad

    class SpyLoss:
        def __init__(self):
            self.last_preds = None
            self.last_targets = None

        def forward(self, preds: np.ndarray, targets: np.ndarray) -> float:
            calls.append(("loss_forward",))
            self.last_preds = preds
            self.last_targets = targets
            # return some arbitrary scalar to check it propagates out of train_batch
            return 0.123

        def backward(self) -> np.ndarray:
            calls.append(("loss_backward",))
            # gradient same shape as preds, simple all-ones for convenience
            return np.ones_like(self.last_preds)

    # Build a small network of 3 spy layers
    layer0 = SpyLayer("L0")
    layer1 = SpyLayer("L1")
    layer2 = SpyLayer("L2")
    layers = [layer0, layer1, layer2]

    loss = SpyLoss()
    nn = NeuralNetwork(layers=layers, loss=loss, seed=42)

    # Fake batch
    x_batch = np.ones((2, 4), dtype=float)
    y_batch = np.zeros_like(x_batch)

    # This should orchestrate: forward over layers, loss forward/backward, then backward over layers
    loss_value = nn.train_batch(x_batch, y_batch)

    # ---- Check call order -------------------------------------------------

    # 3 layer forwards
    assert calls[0:3] == [
        ("forward", "L0"),
        ("forward", "L1"),
        ("forward", "L2"),
    ]

    # loss forward, then loss backward
    assert calls[3] == ("loss_forward",)
    assert calls[4] == ("loss_backward",)

    # 3 layer backwards, in reverse order
    assert calls[5:] == [
        ("backward", "L2"),
        ("backward", "L1"),
        ("backward", "L0"),
    ]

    # ---- Sanity checks on data flow / shapes ------------------------------

    # Forward path is identity through all layers, so preds == x_batch
    np.testing.assert_array_equal(loss.last_preds, x_batch)
    np.testing.assert_array_equal(layer0.last_forward_input, x_batch)
    np.testing.assert_array_equal(layer1.last_forward_input, x_batch)
    np.testing.assert_array_equal(layer2.last_forward_input, x_batch)

    # Backward path: loss.backward returns ones_like(preds); identity grads keep shape
    assert layer2.last_backward_grad.shape == x_batch.shape
    assert layer1.last_backward_grad.shape == x_batch.shape
    assert layer0.last_backward_grad.shape == x_batch.shape

    # train_batch should return the scalar from loss.forward
    assert loss_value == 0.123
