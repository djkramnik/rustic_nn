import numpy as np

def pad(arr, size = 1):
  padded_originals = [np.concatenate([np.zeros(size), row, np.zeros(size)]) for row in arr]
  rows = np.zeros((size, arr.shape[1] + (size * 2)))
  return np.concatenate([rows, padded_originals, rows], axis=0)


def sliding_window(inp: np.ndarray, window_shape: tuple[int, int]):
  [p0, p1] = window_shape
  print(inp)
  print(p0, p1)
  print(inp.strides)


a = np.array([1, 2, 3, 4]).reshape((2,2))
b = np.array([1, 2, 3, 2, 3, 1, 3, 1, 2]).reshape((3, 3))
padded_inp = pad(a, 1)
sliding_window(padded_inp, (3,3))
