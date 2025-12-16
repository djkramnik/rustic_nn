import numpy as np
from util.one_hot_encode import one_hot_encode

def test_one_hot_encode():
  targets = np.array([0, 1, 2, 3])
  num_labels = 4
  encoded = one_hot_encode(targets, num_labels)



  assert(np.allclose(encoded, np.array([
    [1.,0.,0.,0.],
    [0.,1.,0.,0.],
    [0.,0.,1.,0.],
    [0.,0.,0.,1.]
  ])))
