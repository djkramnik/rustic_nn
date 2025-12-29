import numpy as np
from operation.operation import Operation
from util.customtypes import *

class Dropout(Operation):
  def __init__(self, keep_prob: float = 0.8):
    super().__init__()
    if not (0.0 < keep_prob <= 1.0):
      raise ValueError('keep_prob must be in (0, 1]')
    self.keep_prob = keep_prob
    self.inference = True
    self.mask = None

  def forward(self, input_: ndarray, **kwargs):
    self.inference = kwargs.get('inference', True) == False
    return super().forward(input_)

  def _output(self) -> ndarray:
    if self.inference:
      # scale at inference time
      return self._input * self.keep_prob
    else:
      # not inference, mask
      self.mask = np.random.binomial(1, self.keep_prob, size=self.input_.shape)
      return self.input_ * self.mask

  def _input_grad(self, output_grad: ndarray) -> ndarray:
    if self.inference:
      return output_grad * self.keep_prob
    else:
      return output_grad * self.mask


