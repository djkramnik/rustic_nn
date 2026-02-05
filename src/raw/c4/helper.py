import numpy as np

def one_hot_encode(labels: np.ndarray, tot_labels: int) -> np.ndarray:
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

def one_hot_encode_fast(labels: np.ndarray, tot_labels: int) -> np.ndarray:
  assert(labels.ndim == 1)

  encoded = np.zeros((labels.shape[0], tot_labels))
  encoded[np.arange(labels.shape[0]), labels] = 1

  return encoded


def scale_input_data(train: np.ndarray, test: np.ndarray) -> list[np.ndarray, np.ndarray]:
  mean = np.mean(train)
  std = np.std(train)
  return (train - mean) / std, (test - mean) / std

def brew_std(x: np.ndarray) -> np.ndarray:
  mean = np.mean(x)
  return np.sqrt(np.sum(np.power(x - mean, 2)) / x.shape[0])