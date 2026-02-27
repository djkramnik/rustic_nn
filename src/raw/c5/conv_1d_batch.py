import numpy as np
from raw.c5.helper import _pad_1d, assert_dim, assert_same_shape
from raw.c5.conv_1d import _input_grad_1d, conv_1d

input_1d_batch = np.array([
  np.arange(0, 7),
  np.arange(1, 8)
])

# print(input_1d_batch)

# pad batch?  the slow way? ok..

def _pad_1d_batch(inp: np.ndarray, num: int) -> np.ndarray:
  outs = [_pad_1d(obs, num) for obs in inp]
  # print(type(outs))
  return np.stack(outs)

# test = _pad_1d_batch(input_1d_batch, 1)
# print(test)

def conv_1d_batch(inp: np.ndarray, param: np.ndarray) -> np.ndarray:
  outs = [conv_1d(obs, param) for obs in inp]
  #print(type(outs))
  return np.stack(outs)

def _conv_grads_1d_batch(
    inp: np.ndarray,
    param: np.ndarray,
    output_grad: np.ndarray = None) -> np.ndarray:

  assert_dim(inp, 2)
  assert_dim(param, 1)

  param_mid = param.shape[0] // 2
  inp_pad = _pad_1d_batch(inp, 1) # pads along the second dim only
  if output_grad is None:
    output_grad = np.ones_like(inp)
  else:
    assert_same_shape(inp, output_grad)

  param_grad = np.zeros_like(param)
  input_pad_grad = np.zeros_like(inp_pad)

  for b in range(inp.shape[0]):
    for o in range(inp.shape[1]):
      for p in range(param.shape[0]):
        param_grad[p] += inp_pad[b][o + p] * output_grad[b][o]
        input_pad_grad[b][o + p] += param[p] * output_grad[b][o]
  return [
    input_pad_grad[:, param_mid:(inp_pad.shape[1] - param_mid)],
    param_grad
  ]


def input_grad_1d_batch(inp: np.ndarray,
                        param: np.ndarray, out_grad: np.ndarray = None) -> np.ndarray:

    batch_size = inp.shape[0]

    if (out_grad is not None):
       assert_same_shape(inp, out_grad)
       grads = [_input_grad_1d(inp[i], param, out_grad[i]) for i in range(batch_size)]
    else:
      grads = [_input_grad_1d(inp[i], param,) for i in range(batch_size)]

    return np.stack(grads)

def param_grad_1d_batch(inp: np.ndarray,
                        param: np.ndarray,
                        output_grad: np.ndarray = None
                        ) -> np.ndarray:

    if (output_grad is None):
      output_grad = np.ones_like(inp)
    else:
       assert_same_shape(output_grad, inp)

    inp_pad = _pad_1d_batch(inp, 1)

    param_grad = np.zeros_like(param)

    for i in range(inp.shape[0]):
        for o in range(inp.shape[1]):
            for p in range(param.shape[0]):
                param_grad[p] += inp_pad[i][o+p] * output_grad[i][o]

    return param_grad
