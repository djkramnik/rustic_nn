import numpy as np

def softmax(x: np.ndarray) -> np.ndarray:
  return np.exp(x) / np.sum(np.exp(x))

def softmax_safe(x: np.ndarray) -> np.ndarray:
  axis = 1 if x.ndim > 1 else None
  max = np.max(x, axis=axis, keepdims=True)
  return np.exp(x - max) / np.sum(np.exp(x - max), axis=axis, keepdims=True)

# It assumes target is one-hot (N, C)
def binary_cross_entropy(softmax_preds: np.ndarray, target: np.ndarray):
  # prevent log(0)
  p = np.clip(softmax_preds, 1e-9, (1 - 1e-9))
  q = target

  loss = (-q* np.log(p)) - ((1 - q) * np.log(1 - p))
  return np.sum(loss) / softmax_preds.shape[0]

# loss is a scalar bro
def softmax_ce(preds: np.ndarray, target: np.ndarray) -> float
  return binary_cross_entropy(softmax_safe(preds), target)
