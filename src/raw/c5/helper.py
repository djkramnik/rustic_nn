import numpy as np

def assert_same_shape(a: np.ndarray, b: np.ndarray):
  assert a.shape == b.shape
  return None

def assert_dim(a: np.ndarray, dim: int):
  assert len(a.shape) == dim
  return None

# np.pad(inp, (num, num))
def _pad_1d(inp: np.ndarray, num: int):
  padding = np.zeros(num, dtype = inp.dtype)
  return np.concatenate([padding, inp, padding])

