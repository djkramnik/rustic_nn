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