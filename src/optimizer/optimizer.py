class Optimizer(object):
  '''
  Base class for a neural network optimizer
  '''
  def __init__(self, lr: float = 0.06):
    self.lr = lr
  def step(self):
    raise NotImplementedError()