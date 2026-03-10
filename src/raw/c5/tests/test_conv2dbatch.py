import numpy as np

from raw.c5.conv_2d import _compute_output_2d, _pad_2d
from raw.c5.homework.conv2d_batch import conv_2d_batch_fwd, pad_2d_batch

np.random.seed(190220)
imgs_2d_batch = np.random.randn(3, 28, 28)
param_2d = np.random.randn(3,3)

def test_conv2d_batch_pad():
  pad_book = _pad_2d(imgs_2d_batch, 1)
  pad_brain = pad_2d_batch(imgs_2d_batch, 1)
  assert np.allclose(pad_book, pad_brain)

def test_conv2d_batch_fwd():
  book_fwd = _compute_output_2d(imgs_2d_batch, param_2d)
  brain_fwd = conv_2d_batch_fwd(imgs_2d_batch, param_2d)
  assert np.allclose(book_fwd, brain_fwd)