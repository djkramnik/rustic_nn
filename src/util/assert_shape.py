from util.customtypes import *

def assert_shape(
    a: ndarray,
    b: ndarray
):
  assert a.shape == b.shape, \
  '''
  shape mismatch! a: {0}, b:{1}
  '''.format(tuple(a.shape), tuple(b.shape))
  return None