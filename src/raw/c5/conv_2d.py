import numpy as np

from raw.c5.conv_1d_batch import _pad_1d_batch
from raw.c5.helper import _pad_1d, assert_dim

np.random.seed(190220)

imgs_2d_batch = np.random.randn(3, 28, 28)
param_2d = np.random.randn(3, 3)

def _pad_2d_obs(inp: np.ndarray, num: int):
  '''
  Input is a 2 dimensional, square, 2d tensor
  '''
  inp_pad = _pad_1d_batch(inp, num)
  other = np.zeros((num, inp.shape[0] + (num * 2)))
  return np.concatenate([other, inp_pad, other])

def _pad_2d(inp: np.ndarray, num: int):
  '''
  Input is a 3 dimensional tensor, first dimension batch size
  '''
  outs = [_pad_2d_obs(obs, num) for obs in inp]
  return np.stack(outs)

# print(imgs_2d_batch.shape)
padded_2d = _pad_2d(imgs_2d_batch, 1)
# print(padded_2d.shape)

# perform convolution on 2d array
# this is not on the batch.. it is applying the convolution window to a 2d array you feel me
def _compute_output_obs_2d(obs: np.ndarray, param: np.ndarray) -> np.ndarray:
  '''
  obs is a 2d square tensor, so is param
  '''
  param_mid = param.shape[0] // 2
  obs_pad = _pad_2d_obs(obs, param_mid)

  out = np.zeros_like(obs)
  # width now?  but previously it was height..
  for o_w in range(out.shape[0]):
     for o_h in range(out.shape[1]):
        for p_w in range(param.shape[0]):
           for p_h in range(param.shape[1]):
              out[o_w][o_h] += param[p_w][p_h] * obs_pad[o_w+p_w][o_h+p_h]
  return out

def _compute_output_2d(img_batch: np.ndarray, param: np.ndarray) -> np.ndarray:
   assert_dim(img_batch, 3)
   outs = [_compute_output_obs_2d(obs, param) for obs in img_batch]
   return np.stack(outs)

print('conv forward output', _compute_output_2d(imgs_2d_batch, param_2d).shape)

type ndarr = np.ndarray

def _compute_grads_obs_2d(input_obs: ndarr, output_grad_obs: ndarr, param: ndarr) -> ndarr:
  '''
  input_obs: 2d tensor representing the input observation (non batch)
  output_grad: 2d tendor representing the output gradient
  param: 2d filter
  '''
  param_size = param.shape[0]
  param_mid = param.shape[0] // 2
  output_obs_pad = _pad_2d_obs(output_grad_obs, param_mid)
  input_grad = np.zeros_like(input_obs)

  for i_w in range(input_obs.shape[0]):
     for i_h in range(input_obs.shape[1]):
        for p_w in range(param_size):
           for p_h in range(param_size):
              input_grad[i_w][i_h] += output_obs_pad[i_w + param_size - p_w - 1][i_h + param_size- p_h - 1] * param[p_w][p_h]
  return input_grad

def _compute_grads_2d(inp: ndarr, output_grad: ndarr, param: ndarr) -> ndarr:
  grads = [_compute_grads_obs_2d(inp[i], output_grad[i], param) for i in range(output_grad.shape[0])]
  return np.stack(grads)

def _param_grad_2d(inp: ndarr, output_grad: ndarr, param: ndarr) -> ndarr:
  param_size = param.shape[0]
  param_mid = param_size // 2
  inp_pad = _pad_2d(inp, param_mid)
  param_grad = np.zeros_like(param)
  img_shape = output_grad.shape[1:] # output_grad is 3d this time.  img_shape is the shape after the batch dim

  # I suppose inp is 3d too, preserving the batch
  for i in range(inp.shape[0]):
    for o_w in range(img_shape[0]):
        for o_h in range(img_shape[1]):
          for p_w in range(param_size):
              for p_h in range(param_size):
                param_grad[p_w][p_h] += inp_pad[i][o_w + p_w][o_h + p_h] * output_grad[i][o_w][o_h]
  return param_grad

def _compute_output_2d_sum(img_batch: ndarr,
                           param: ndarr):

    out = _compute_output_2d(img_batch, param)

    return out.sum()


