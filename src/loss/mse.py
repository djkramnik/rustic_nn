import numpy as np
from util.customtypes import *
from util.assert_shape import *
from loss.loss import Loss

class MeanSquaredError(Loss):
  def _output(self):
    # per chat gpt
    # return np.mean(np.power(self.prediction - self.target, 2))
    return np.sum(np.power(self.prediction - self.target, 2)) / self.prediction.shape[0]

  def _input_grad(self):
    return 2 * (self.prediction - self.target) / self.prediction.shape[0]
