import numpy as np

a = np.array([1, 2, 3, 4]).reshape((2,2))

b = np.array([1, 2, 3, 2, 3, 1, 3, 1, 2]).reshape((3, 3))

def pad(arr, size = 1):
  padded_originals = [np.concatenate([np.zeros(size), row, np.zeros(size)]) for row in arr]
  rows = np.zeros((size, arr.shape[1] + (size * 2)))
  return np.concatenate([rows, padded_originals, rows], axis=0)

def conv_squares(arr, param):
  pw = param.shape[0]
  pad_sz = pw // 2
  padded_inp = pad(arr, pad_sz)
  output = np.zeros_like(arr)
  for i in range(arr.shape[0]):
    for j in range(arr.shape[1]):
      window = padded_inp[i:i+pw,j:j+pw]
      output[i][j] = np.sum(window * param)
  return output

print(conv_squares(a, b))