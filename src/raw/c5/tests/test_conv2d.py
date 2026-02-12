import numpy as np

from raw.c5.conv_2d import _compute_grads_2d, _compute_output_2d_sum, _param_grad_2d

def test_batch_2d_back():
  np.random.seed(190220)

  imgs_2d_batch = np.random.randn(3, 28, 28)
  param_2d = np.random.randn(3, 3)
  img_grads = _compute_grads_2d(imgs_2d_batch,
                              np.ones_like(imgs_2d_batch),
                              param_2d)
  assert np.allclose(img_grads.shape, imgs_2d_batch.shape)
  print(img_grads.shape)
  param_grad = _param_grad_2d(imgs_2d_batch,
                              np.ones_like(imgs_2d_batch),
                              param_2d)
  assert np.allclose(param_2d.shape, param_grad.shape)
  print(param_grad.shape)

  imgs_2d_batch_2 = imgs_2d_batch.copy()
  imgs_2d_batch_2[0][6][18] += 1

  output_diff = _compute_output_2d_sum(imgs_2d_batch_2, param_2d) - _compute_output_2d_sum(imgs_2d_batch, param_2d)
  assert np.allclose(img_grads[0][6][18], output_diff)
  print(output_diff)

  param_2d_2 = param_2d.copy()
  param_2d_2[0][2] += 1
  output_diff2 = _compute_output_2d_sum(imgs_2d_batch, param_2d_2) - _compute_output_2d_sum(imgs_2d_batch, param_2d)
  assert np.allclose(param_grad[0][2], output_diff2)
  print(output_diff2)




