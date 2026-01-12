import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
  return 1 / (1 + np.exp(x * -1))

def sigmoid_deriv(x: np.ndarray) -> np.ndarray:
  return np.exp(x * -1) / (np.power(1 + np.exp(x * -1), 2))

def sigmoid_deriv_elegant(x: np.ndarray) -> np.ndarray:
  return sigmoid(x) * (1 - sigmoid(x))

