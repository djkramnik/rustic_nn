import numpy as np
from util.customtypes import ndarray

def one_hot_encode(targets: ndarray, num_labels: int):
  # create an array where for each row of targets we have num_label columns, all set to 0
  rows = targets.shape[0]
  encoded = np.zeros((rows, num_labels))
  # the non encoded target values are 0 based index
  for i in range(rows):
    encoded[i][targets[i]] = 1
  return encoded
