# test_dense.py

import numpy as np

from operation.operation import Operation
import layer.dense as dense_module  # this is where Dense, WeightMultiply, BiasAdd live


def test_dense_forward_and_backward_call_order(monkeypatch):
    """
    Verify that:
    - Dense.forward calls its operations in order: weight -> bias -> activation
    - Dense.backward calls them in reverse: activation -> bias -> weight
    using spy Operations instead of the real math ops.
    """
    calls = []  # we'll record ("forward" | "backward", name) tuples here

    class SpyOperation(Operation):
        def __init__(self, name: str):
            super().__init__()
            self.name = name

        def _output(self) -> np.ndarray:
            # identity op: output == input
            return self.input_

        def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
            # identity gradient: pass grad through unchanged
            return output_grad

        def forward(self, input_: np.ndarray) -> np.ndarray:
            calls.append(("forward", self.name))
            return super().forward(input_)

        def backward(self, output_grad: np.ndarray) -> np.ndarray:
            calls.append(("backward", self.name))
            return super().backward(output_grad)

    # Activation class to plug into Dense(…, activation=...)
    class FakeActivation(SpyOperation):
        def __init__(self):
            # no args: matches Dense's expectation that activation() takes no params
            super().__init__("activation")

    # Fake WeightMultiply / BiasAdd factories that return spy ops
    def fake_weight_multiply(param):
        # param is the weight matrix; we ignore it for this spy
        return SpyOperation("weight")

    def fake_bias_add(param):
        # param is the bias vector; we ignore it for this spy
        return SpyOperation("bias")

    # Patch the names used *inside* layer.dense
    monkeypatch.setattr(dense_module, "WeightMultiply", fake_weight_multiply)
    monkeypatch.setattr(dense_module, "BiasAdd", fake_bias_add)

    Dense = dense_module.Dense

    # Instantiate a Dense layer with our fake activation class
    layer = Dense(neurons=3, activation=FakeActivation())

    x = np.ones((2, 4), dtype=float)

    # Forward pass: should chain Spy(weight) -> Spy(bias) -> FakeActivation
    out = layer.forward(x)

    # Backward pass: gradient same shape as out so Layer/Operation shape asserts are happy
    grad_out = np.ones_like(out)
    layer.backward(grad_out)

    # Check call order:
    # forward: weight -> bias -> activation
    assert calls[:3] == [
        ("forward", "weight"),
        ("forward", "bias"),
        ("forward", "activation"),
    ]

    # backward: activation -> bias -> weight
    assert calls[3:] == [
        ("backward", "activation"),
        ("backward", "bias"),
        ("backward", "weight"),
    ]

    # Optional sanity checks: the spies are identity ops, so:
    assert np.array_equal(out, x)
    # and the final gradient returned by backward should be equal to grad_out
    # (this uses how your Operation/Layer.backward is implemented, but is true
    #  for identity spies in a purely linear chain)
