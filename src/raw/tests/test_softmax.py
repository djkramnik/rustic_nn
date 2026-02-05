import numpy as np
from raw.c4.loss import softmax_safe, binary_cross_entropy

def test_bce():
  a = np.array([2, 7, 1])
  b = np.array([0, 1, 0])
  preds = softmax_safe(a)
  print(f'softmax: {preds}')
  loss = binary_cross_entropy(preds, b)
  print(f'binary_cross_entropy: {loss}')
