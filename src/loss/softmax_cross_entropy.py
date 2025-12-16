import numpy as np
from loss.loss import Loss
from util.np_utils import normalize, unnormalize, softmax

class SoftmaxCrossEntropy(Loss):
  def __init__(self, eps:float = 1e-9):
    super().__init__()
    self.eps = eps

  def _output(self) -> float:
    softmax_preds = softmax(self.prediction, axis=1)

    # clipping the softmax output to prevent numeric instability?!
    self.softmax_preds = np.clip(softmax_preds, self.eps, 1 - self.eps)

    # actual loss computation
    softmax_cross_entropy_loss = -1.0 * self.target * np.log(self.softmax_preds) - (
        1.0 - self.target
    ) * np.log(1 - self.softmax_preds)

    return np.sum(softmax_cross_entropy_loss) / self.prediction.shape[0]

  def _input_grad(self) -> np.ndarray:
    return (self.softmax_preds - self.target) / self.prediction.shape[0]
