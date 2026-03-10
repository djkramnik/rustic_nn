import numpy as np

from raw.c5.conv_2d import _compute_grads_2d, _compute_grads_obs_2d, _compute_output_2d_sum, _compute_output_obs_2d, _param_grad_2d
from raw.c5.homework.conv2d import conv_2d_fwd, conv_2d_inp_grad, conv_2d_param_grad

def test_conv2d_fwd():
  inp = np.arange(16).reshape((4,4)) + 1
  param = np.arange(9).reshape((3,3)) + 1
  book_fwd = _compute_output_obs_2d(inp, param)
  homework_fwd = conv_2d_fwd(inp, param)
  assert np.allclose(book_fwd, homework_fwd)

def test_conv2d_back():
  inp = np.arange(16).reshape((4,4)) + 1
  param = np.arange(9).reshape((3,3)) + 1
  book_inp_grad = _compute_grads_obs_2d(inp, np.ones_like(inp), param)
  brain_inp_grad = conv_2d_inp_grad(inp, param)
  # print(book_inp_grad)
  # print(brain_inp_grad)
  assert np.allclose(book_inp_grad, brain_inp_grad)
  evil_out = inp - 1
  book_inp_grad2 = _compute_grads_obs_2d(inp, evil_out, param)
  brain_inp_grad2 = conv_2d_inp_grad(inp, param, evil_out)
  # print(book_inp_grad2)
  # print(brain_inp_grad2)
  assert np.allclose(book_inp_grad2, brain_inp_grad2)

def test_conv2d_param_grad():
  inp = np.arange(16).reshape((4,4)) + 1
  param = np.arange(9).reshape((3,3)) + 1
  evil_out = inp - 1

  batched_inp = np.expand_dims(inp, axis=0)
  book_param_grad = _param_grad_2d(batched_inp, np.ones_like(batched_inp), param)
  brain_param_grad = conv_2d_param_grad(inp, param)

  assert np.allclose(book_param_grad, brain_param_grad)

  book_param_grad2 = _param_grad_2d(batched_inp, np.expand_dims(evil_out, axis=0), param)
  brain_param_grad2 = conv_2d_param_grad(inp, param, evil_out)
  assert np.allclose(book_param_grad2, brain_param_grad2)


def test_batch_2d_back():
  np.random.seed(190220)

  imgs_2d_batch = np.random.randn(3, 28, 28)
  param_2d = np.random.randn(3, 3)
  img_grads = _compute_grads_2d(imgs_2d_batch,
                              np.ones_like(imgs_2d_batch),
                              param_2d)
  assert np.allclose(img_grads.shape, imgs_2d_batch.shape)
  # print(img_grads.shape)
  param_grad = _param_grad_2d(imgs_2d_batch,
                              np.ones_like(imgs_2d_batch),
                              param_2d)
  assert np.allclose(param_2d.shape, param_grad.shape)
  print(param_grad.shape)

  imgs_2d_batch_2 = imgs_2d_batch.copy()
  imgs_2d_batch_2[0][6][18] += 1

  output_diff = _compute_output_2d_sum(imgs_2d_batch_2, param_2d) - _compute_output_2d_sum(imgs_2d_batch, param_2d)
  assert np.allclose(img_grads[0][6][18], output_diff)
  # print(output_diff)

  param_2d_2 = param_2d.copy()
  param_2d_2[0][2] += 1
  output_diff2 = _compute_output_2d_sum(imgs_2d_batch, param_2d_2) - _compute_output_2d_sum(imgs_2d_batch, param_2d)
  assert np.allclose(param_grad[0][2], output_diff2)
  # print(output_diff2)




