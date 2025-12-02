from util.customtypes import *
from util.assert_shape import *

class Loss(object):
  def __init__(self):
    pass
  def forward(self, preds: ndarray, target: ndarray) -> ndarray:
    assert_shape(preds, target)
    self.prediction = preds
    self.target = target
    loss_value = self._output()
    return loss_value
  def backward(self):
    '''
    Computes gradient of the loss value wrt the input of the loss function
    No grad param here like in other ops.. because this is the source of the back grads!
    '''
    self.input_grad = self._input_grad()
    assert_shape(self.prediction, self.input_grad)
    return self.input_grad

  def _output(self):
    raise NotImplementedError()
  def _input_grad(self) -> ndarray:
    raise NotImplementedError()

