import numpy as np
from util.customtypes import *
from util.assert_shape import assert_shape
from operation.operation import Operation

class Linear(Operation):
  '''
  "Identity" activation function
  '''
  def _output(self) -> ndarray:
    '''
    Pass through the input
    '''
    return self.input_
  def _input_grad(self, output_grad: ndarray) -> ndarray:
    '''
    Pass through the output grad
    '''
    return output_grad

class Sigmoid(Operation):
  def _output(self) -> ndarray:
    return 1.0 / (1.0 + np.exp(-1.0 * self.input_))

  def _input_grad(self, output_grad: ndarray) -> ndarray:
    sigmoid_deriv = self.output * (1.0 - self.output)
    return sigmoid_deriv * output_grad

