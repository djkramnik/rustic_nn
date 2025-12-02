from util.customtypes import *
from util.assert_shape import *

class Operation(object):
  '''
  Base class for an operation in a neural network
  '''
  def __init__(self):
    pass
  def forward(self, input_: ndarray):
    self.input_ = input_
    self.output = self._output()
    return self.output
  def backward(self, output_grad: ndarray) -> ndarray:
    '''
    calls self._input_grad() with impl specific backprop
    '''
    assert_shape(self.output, output_grad)
    self.input_grad = self._input_grad(output_grad)
    assert_shape(self.input_grad, self.input_)
    return self.input_grad
  def _output(self) -> ndarray:
    '''
    The actual operation of forward
    '''
    raise NotImplementedError()
  def _input_grad(self, output_grad) -> ndarray:
    raise NotImplementedError()

class ParamOperation(Operation):
  def __init__(self, param: ndarray):
    super().__init__()
    self.param = param
  def backward(self, output_grad: ndarray):
    assert_shape(self.output, output_grad)
    self.input_grad = self._input_grad(output_grad)
    self.param_grad = self._param_grad(output_grad)
    assert_shape(self.param, self.param_grad)
    assert_shape(self.input_, self.input_grad)

    return self.input_grad # why?
  def _param_grad(self, output_grad:  ndarray) -> ndarray:
    '''
    Calculate the gradient of the output wrt the param,
    and then multiply by output_grad to get the deriv of the
    loss wrt the param
    '''
    raise NotImplementedError()