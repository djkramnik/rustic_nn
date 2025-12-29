import numpy as np

def one_hot_encode(labels: np.ndarray, tot_labels: int):
  # this just has rows and no second dim
  assert(labels.ndim == 1)
  encoded = np.zeros((labels.shape[0], tot_labels))

  for index, label in enumerate(labels):
    encoded[index][label] = 1

  return encoded


# intuition around vectorization

# mask = np.zeros((3, 3))
# toy_labels = np.array([0, 2, 1])
# print(toy_labels.shape)
# mask[np.array([0, 1, 2]), toy_labels] = 1
# print(mask)

def one_hot_encode_fast(labels: np.ndarray, tot_labels: int):
  assert(labels.ndim == 1)

  encoded = np.zeros((labels.shape[0], tot_labels))
  encoded[np.arange(labels.shape[0]), labels] = 1

  return encoded


