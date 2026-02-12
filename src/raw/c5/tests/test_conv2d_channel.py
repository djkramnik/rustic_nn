import numpy as np

from raw.c5.conv_2d_channels import _compute_output_sum, _input_grad, _param_grad

np.random.seed(190220)

def test_conv_2d_chan():
  cifar_imgs = np.random.randn(10, 3, 32, 32)
  cifar_param = np.random.randn(3, 16, 5, 5)

  test_idx = (3, 1, 2, 19)
  cifar_imgs_2 = cifar_imgs.copy()
  cifar_imgs_2[test_idx] += 1

  unaltered_sum = _compute_output_sum(cifar_imgs, cifar_param)
  output_diff = _compute_output_sum(cifar_imgs_2, cifar_param) - unaltered_sum

  output_grad = np.ones((10, 16, 32, 32))
  input_grad = _input_grad(cifar_imgs, output_grad, cifar_param)
  assert np.allclose(output_diff, input_grad[test_idx])
  print('finished input_grad test')
  test_p_idx = (0,8,0,2)
  cifar_param_2 = cifar_param.copy()
  cifar_param_2[test_p_idx] += 1

  output_diff2 = _compute_output_sum(cifar_imgs, cifar_param_2) - unaltered_sum
  param_grad = _param_grad(cifar_imgs, np.ones((10,16,32,32)), cifar_param)
  assert np.allclose(output_diff2, param_grad[test_p_idx])