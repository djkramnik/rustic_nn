from optimizer.optimizer import Optimizer

class SGD(Optimizer):
  # this override just uses a different default LR than Optimizer
  def __init__(self, lr: float = 0.01):
    super().__init__(lr)
  def step(self):
    '''
    for each parameter,
    adjust in the appropriate direction,
    with the magnitude of the adjustment based on the LR
    '''
    for(param, param_grad) in zip(self.net.params(), self.net.param_grads()):
      param -= self.lr * param_grad