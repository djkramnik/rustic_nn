import numpy as np

def one_hot_encode(labels: np.ndarray, tot_labels: int):
  # this just has rows and no second dim

  assert(labels.ndim == 1)
