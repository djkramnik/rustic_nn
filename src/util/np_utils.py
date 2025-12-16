from scipy import special
import numpy as np

def normalize(a: np.ndarray):
    other = 1 - a
    return np.concatenate([a, other], axis=1)

def unnormalize(a: np.ndarray):
    return a[np.newaxis, 0]

def softmax(x: np.ndarray, axis=None) -> np.ndarray:
    return np.exp(x - special.logsumexp(x, axis=axis, keepdims=True))