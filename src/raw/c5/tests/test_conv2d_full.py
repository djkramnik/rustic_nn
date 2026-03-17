import numpy as np

from raw.c5.conv_2d_channels import _output
from raw.c5.homework.conv2d_full import conv_2d_full

np.random.seed(190220)
cifar_imgs = np.random.randn(10, 3, 32, 32)
cifar_param = np.random.randn(3, 16, 5, 5)

# def test_fwd():
#   output_brain = conv_2d_full(cifar_imgs, cifar_param)
#   book_brain = _output(cifar_imgs, cifar_param)
#   assert np.allclose(output_brain, book_brain)

def test_fwd_lite():
  batcha = np.arange(1, 17).reshape((4,4))
  batchb = np.arange(2, 18).reshape((4,4))
  c = np.expand_dims(np.stack((batcha, batchb)), axis=1)
  assert np.allclose(c.shape, (2, 1, 4, 4))

  parama = np.arange(9).reshape((3,3))
  paramb = np.arange(1,10).reshape((3,3))
  params = np.expand_dims(np.stack((parama, paramb)), axis=0)
  assert np.allclose(params.shape, (1, 2, 3, 3))

  output = _output(c, params)
  output_brain = conv_2d_full(c, params)
  assert np.allclose(output, output_brain)
  assert np.allclose(output_brain.shape, (2, 2, 4, 4))
  assert np.allclose(output_brain[0][0][0][0], 97)
  assert np.allclose(output_brain[0][1][0][0], 111)
  assert np.allclose(output_brain[1][0][0][0], 121)
  assert np.allclose(output_brain[1][1][0][0], 139)

