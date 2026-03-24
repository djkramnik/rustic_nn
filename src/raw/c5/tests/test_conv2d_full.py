import numpy as np

from raw.c5.conv_2d_channels import _input_grad, _output
from raw.c5.homework.conv2d import conv_2d_fwd
from raw.c5.homework.conv2d_full import conv_2d_full, full_inp_grad

np.random.seed(190220)
cifar_imgs = np.random.randn(10, 3, 32, 32)
cifar_param = np.random.randn(3, 16, 5, 5)

def test_inp_grad():
  neutral_outgrad = np.ones((cifar_imgs.shape[0], cifar_param.shape[1], cifar_imgs.shape[2], cifar_imgs.shape[3]))
  print(neutral_outgrad.shape)
  book = _input_grad(cifar_imgs, neutral_outgrad, cifar_param)
  brain = full_inp_grad(cifar_imgs, cifar_param)
  assert np.allclose(book, brain)

  evil_outgrad = np.random.randn(cifar_imgs.shape[0], cifar_param.shape[1], cifar_imgs.shape[2], cifar_imgs.shape[3])
  print(evil_outgrad.shape)
  book2 = _input_grad(cifar_imgs, evil_outgrad, cifar_param)
  brain2 = full_inp_grad(cifar_imgs, cifar_param, evil_outgrad)
  assert np.allclose(book2, brain2)

# def test_secondlayer():
#   param2 = np.array([
#     [
#       [
#         [1, 1, 1],
#         [1, 3, 3],
#         [1, 3, 3]
#       ],
#       [
#         [3, 1, 1],
#         [1, 2, 2],
#         [2, 2, 3]
#       ]
#     ],
#     [
#       [
#         [3, 1, 3],
#         [3, 1, 2],
#         [3, 2, 3]
#       ],
#       [
#         [2, 3, 2],
#         [3, 2, 3],
#         [1, 3, 1]
#       ]
#     ],
#     [
#       [
#         [2, 3, 2],
#         [3, 2, 3],
#         [1, 2, 1]
#       ],
#       [
#         [1, 3, 3],
#         [1, 1, 1],
#         [1, 2, 1]
#       ]
#     ],
#   ])
#   homework = np.array([
#     [
#       [
#         [169, 149],
#         [154, 198]
#       ],
#       [
#         [172, 139],
#         [164, 91]
#       ],
#       [
#         [105, 66],
#         [143, 77]
#       ]
#     ]
#   ])
#   allegedo2 = np.array([
#     [
#       [3832, 3437],
#       [3273, 3295]
#     ],
#     [
#       [3416, 2870],
#       [3150, 3164]
#     ]
#   ])
#   bookish = conv_2d_full(homework, param2)
#   DAbook = _output(homework, param2)
#   print(DAbook)
#   assert np.allclose(allegedo2, bookish, DAbook)

# def test_firstlayer():
#   inp = np.array([
#     [
#       [
#         [1, 8],
#         [2, 5]
#       ],
#       [
#         [2, 5],
#         [7, 4]
#       ],
#     ]
#   ])
#   param = np.array([
#     [
#       [
#         [3, 7, 4],
#         [3, 1, 0],
#         [1, 6, 7]
#       ],
#       [
#         [7, 1, 5],
#         [1, 8, 6],
#         [2, 4, 2],
#       ],
#       [
#         [3, 2, 7],
#         [3, 1, 2],
#         [5, 0, 7]
#       ]
#     ],
#     [
#       [
#         [4, 8, 7],
#         [8, 6, 5],
#         [4, 8, 7]
#       ],
#       [
#         [2, 3, 5],
#         [1, 2, 8],
#         [2, 6, 3]
#       ],
#       [
#         [1, 1, 6],
#         [4, 3, 5],
#         [2, 2, 2]
#       ]
#     ]
#   ])
#   homework = np.array([
#     [
#       [
#         [169, 149],
#         [154, 198]
#       ],
#       [
#         [172, 139],
#         [164, 91]
#       ],
#       [
#         [105, 66],
#         [143, 77]
#       ]
#     ]
#   ])
#   bookish = conv_2d_full(inp, param)
#   assert np.allclose(homework, bookish)
#   print(bookish)
#   # output_first = conv_2d_full()

# def test_fwd():
#   output_brain = conv_2d_full(cifar_imgs, cifar_param)
#   book_brain = _output(cifar_imgs, cifar_param)
#   assert np.allclose(output_brain, book_brain)

# def test_fwd_lite():
#   batcha = np.arange(1, 17).reshape((4,4))
#   batchb = np.arange(2, 18).reshape((4,4))
#   c = np.expand_dims(np.stack((batcha, batchb)), axis=1)
#   assert np.allclose(c.shape, (2, 1, 4, 4))

#   parama = np.arange(9).reshape((3,3))
#   paramb = np.arange(1,10).reshape((3,3))
#   params = np.expand_dims(np.stack((parama, paramb)), axis=0)
#   assert np.allclose(params.shape, (1, 2, 3, 3))

#   output = _output(c, params)
#   output_brain = conv_2d_full(c, params)
#   assert np.allclose(output, output_brain)
#   assert np.allclose(output_brain.shape, (2, 2, 4, 4))
#   assert np.allclose(output_brain[0][0][0][0], 97)
#   assert np.allclose(output_brain[0][1][0][0], 111)
#   assert np.allclose(output_brain[1][0][0][0], 121)
#   assert np.allclose(output_brain[1][1][0][0], 139)

