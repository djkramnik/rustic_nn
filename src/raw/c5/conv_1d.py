import numpy as np
from raw.c5.helper import assert_dim, assert_same_shape, _pad_1d
# forward 1d convolution..

def conv_1d(inp: np.ndarray, param: np.ndarray) -> np.ndarray:
  assert_dim(inp, 1)
  assert_dim(param, 1)
  # pad input based on param length
  # param_mid = param.shape[0] // 2
  inp_pad = _pad_1d(inp, param.shape[0] // 2)
  # initialize output.  it is the same shape as inp here because we are not downsampling
  out = np.zeros(inp.shape)

  # the actual convolution
  for o in range(len(out)):
    inp_win = inp_pad[o:o+len(param)]
    out[o] = np.dot(inp_win, param)

  assert_same_shape(inp, out)
  return out

def _conv_grads_1d(
    inp: np.ndarray,
    param: np.ndarray,
    output_grad: np.ndarray = None) -> np.ndarray:
  assert_dim(inp, 1)
  assert_dim(param, 1)
  param_mid = param.shape[0] // 2
  inp_pad = _pad_1d(inp, param_mid)

  if output_grad is None:
    output_grad = np.ones_like(inp)
  else:
    assert_same_shape(inp, output_grad)

  param_grad = np.zeros_like(param)
  input_pad_grad = np.zeros_like(inp_pad)

  for o in range(inp.shape[0]):
    for p in range(param.shape[0]):
      param_grad[p] += inp_pad[o + p] * output_grad[o]
      # todo: revisit why I got this wrong originally? that is, why it needs to be in the loop
      input_pad_grad[o + p] += param[p] * output_grad[o]

  return [input_pad_grad[param_mid:(inp_pad.shape[0]-param_mid)], param_grad]

def _input_grad_1d(inp: np.ndarray,
                   param: np.ndarray,
                   output_grad: np.ndarray = None) -> np.ndarray:

    param_len = param.shape[0]
    param_mid = param_len // 2

    if output_grad is None:
        output_grad = np.ones_like(inp)
    else:
        assert_same_shape(inp, output_grad)

    output_pad = _pad_1d(output_grad, param_mid)

    # Zero padded 1 dimensional convolution
    param_grad = np.zeros_like(param)
    input_grad = np.zeros_like(inp)

    for o in range(inp.shape[0]):
        for f in range(param.shape[0]):
            input_grad[o] += output_pad[o+param_len-f-1] * param[f]

    assert_same_shape(param_grad, param)

    return input_grad

def _param_grad_1d(inp: np.ndarray,
                   param: np.ndarray,
                   output_grad: np.ndarray = None) -> np.ndarray:

    param_len = param.shape[0]
    param_mid = param_len // 2
    input_pad = _pad_1d(inp, param_mid)

    if output_grad is None:
        output_grad = np.ones_like(inp)
    else:
        assert_same_shape(inp, output_grad)

    # Zero padded 1 dimensional convolution
    param_grad = np.zeros_like(param)
    input_grad = np.zeros_like(inp)

    for o in range(inp.shape[0]):
        for p in range(param.shape[0]):
            param_grad[p] += input_pad[o+p] * output_grad[o]

    assert_same_shape(param_grad, param)

    return param_grad
