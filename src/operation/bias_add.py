import numpy as np
from util.customtypes import *
from operation.operation import ParamOperation

class BiasAdd(ParamOperation):
  # all bias add does is elementwise add itself to the input
  def _output(self) -> ndarray:
    return self.input_ + self.param

  # a noop?
  def _input_grad(self, output_grad: ndarray) -> ndarray:
    return output_grad * np.ones_like(self.input_)

  def _param_grad(self, output_grad: ndarray) -> ndarray:
    param_grad = output_grad * np.ones_like(self.param)
    return np.sum(param_grad, axis=0).reshape(1, param_grad.shape[1])

