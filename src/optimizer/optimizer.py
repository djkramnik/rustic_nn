import numpy as np

class Optimizer(object):
  '''
  Base class for a neural network optimizer
  '''
  def __init__(self, lr: float = 0.01, final_lr: float = 0, decay_type: str = None) -> None:
    self.lr = lr
    self.final_lr = final_lr
    self.decay_type = decay_type
    self.first = True

  def _setup_decay(self) -> None:
    if not self.decay_type:
      return
    elif self.decay_type == "exponential":
      self.decay_per_epoch = np.power(self.final_lr / self.lr, 1.0 / (self.max_epochs - 1))
    elif self.decay_type == "linear":
      self.decay_per_epoch = (self.lr - self.final_lr) / (self.max_epochs - 1)

  def step(self) -> None:
    for param, param_grad in zip(self.net.params(), self.net.param_grads()):
      self._update_rule(param=param, grad=param_grad)

  def _update_rule(self, **kwargs) -> None:
    raise NotImplementedError()