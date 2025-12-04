from layer.layer import Layer
from operation.activation import Sigmoid
from operation.operation import Operation
from operation.weightmultiply import WeightMultiply
from operation.bias_add import BiasAdd
from util.customtypes import *
from typing import Type
import numpy as np
import numbers

class Dense(Layer):
  '''
  A fully connected layer that inherits from Layer
  '''
  def __init__(self, neurons: int, activation: Type[Operation] = Sigmoid):
    super().__init__(neurons)
    self.activation = activation()

  def _setup_layer(self, input_: ndarray):
    # thank you benefactor?
    if hasattr(self, "seed") and isinstance(self.seed, numbers.Integral):
      np.random.seed(int(self.seed))

    # this will get wiped out by calling _params
    self.params = []

    # matmul weight param is (#features, hidden_size)
    # should we not normalize the weights?
    self.params.append(
      np.random.randn(input_.shape[1], self.neurons)
    )
    # bias param
    # bias weights are (1, hidden_size)
    self.params.append(np.random.randn(1, self.neurons))

    self.operations = [
      WeightMultiply(self.params[0]),
      BiasAdd(self.params[1]),
      self.activation
    ]


