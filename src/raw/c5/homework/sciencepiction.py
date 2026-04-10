import numpy as np
from scipy.signal import correlate2d

a = np.arange(1, 13).reshape((3, 4))
b = np.array([
    1, 2, 2, 1, 1,
    1, 1, 1, 2, 2,
    2, 2, 1, 2, 2,
    1, 1, 2, 1, 1,
    2, 1, 1, 2, 2
]).reshape((5, 5))

out = correlate2d(a, b, mode="same", boundary="fill", fillvalue=0)
print(out)