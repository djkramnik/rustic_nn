import numpy as np

def softmax(x: np.ndarray) -> np.ndarray:
  return np.exp(x) / np.sum(np.exp(x))

def softmax_safe(x: np.ndarray) -> np.ndarray:
  axis = 1 if x.ndim > 1 else None
  max = np.max(x, axis=axis, keepdims=True)
  return np.exp(x - max) / np.sum(np.exp(x - max), axis=axis, keepdims=True)


# impl this next please
def binary_cross_entropy(preds: np.ndarray, target: np.ndarray):
  # apply softmax in here
  pass