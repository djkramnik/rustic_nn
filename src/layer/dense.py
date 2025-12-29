from layer.layer import Layer
from operation.activation import Sigmoid
from operation.operation import Operation
from operation.weightmultiply import WeightMultiply
from operation.bias_add import BiasAdd
from operation.dropout import Dropout
from util.customtypes import *
from typing import Type
import numpy as np
import numbers

class Dense(Layer):
  '''
  A fully connected layer that inherits from Layer
  '''
  def __init__(self,
      neurons: int,
      activation: Operation = Sigmoid(),
      weight_init: str = None,
      dropout: float | None = None
    ):
    super().__init__(neurons)
    self.weight_init = weight_init
    self.activation = activation
    self.dropout = dropout

  def _setup_layer(self, input_: ndarray):
    # thank you benefactor?
    if hasattr(self, "seed") and isinstance(self.seed, numbers.Integral):
      np.random.seed(int(self.seed))

    # this will get wiped out by calling _params
    self.params = []

    # matmul weight param is (#features, hidden_size)
    # should we not normalize the weights?

    fan_in = input_.shape[1]

    # weight initialization.  right now just glorot or nothing
    scale = 1 if self.weight_init != 'glorot' else np.sqrt(2 / (fan_in + self.neurons))

    weights = np.random.randn(fan_in, self.neurons) * scale
    # bias weights are (1, hidden_size)
    # bias = np.random.randn(1, self.neurons) * scale

    # test setting bias to zero
    bias = np.zeros((1, self.neurons))

    self.params.append(weights)
    self.params.append(bias)

    self.operations = [
      WeightMultiply(weights),
      BiasAdd(bias),
      self.activation
    ]
    if self.dropout is not None:
      self.operations.append(Dropout(keep_prob=self.dropout))


