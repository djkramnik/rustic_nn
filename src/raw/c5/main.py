import numpy as np
from raw.c5.helper import _pad_1d, assert_same_shape, assert_dim

# 1D Convolution

input_1d = np.array([1, 2, 3, 4, 5])
param_1d = np.array([1, 1, 1])

print(_pad_1d(input_1d, 1))