import numpy as np

from raw.c5.conv_2d import _compute_grads_2d, _compute_output_2d, _pad_2d, _param_grad_2d
from raw.c5.homework.conv2d_batch import conv_2d_batch_fwd, conv_2d_batch_inpgrad, conv_2d_batch_paramgrad, pad_2d_batch

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

def test_conv2d_batch_inpgrad():
  outgrad = np.ones_like(imgs_2d_batch)
  book_inp_grad = _compute_grads_2d(imgs_2d_batch, outgrad, param_2d)
  brain_inp_grad = conv_2d_batch_inpgrad(imgs_2d_batch, param_2d, outgrad)
  assert np.allclose(book_inp_grad, brain_inp_grad)
  outgrad2 = imgs_2d_batch + 1
  book_inp_grad2 = _compute_grads_2d(imgs_2d_batch, outgrad2, param_2d)
  brain_inp_grad2 = conv_2d_batch_inpgrad(imgs_2d_batch, param_2d, outgrad2)
  assert np.allclose(book_inp_grad2, brain_inp_grad2)

def test_conv2d_batch_paramgrad():
  outgrad = np.ones_like(imgs_2d_batch)
  book_param_grad = _param_grad_2d(imgs_2d_batch, outgrad, param_2d)
  brain_param_grad = conv_2d_batch_paramgrad(imgs_2d_batch, param_2d, outgrad)
  assert np.allclose(book_param_grad, brain_param_grad)

  outgrad2 = imgs_2d_batch + 1
  book_param_grad2 = _param_grad_2d(imgs_2d_batch, outgrad2, param_2d)
  brain_param_grad2 = conv_2d_batch_paramgrad(imgs_2d_batch, param_2d, outgrad2)
  assert np.allclose(book_param_grad2, brain_param_grad2)