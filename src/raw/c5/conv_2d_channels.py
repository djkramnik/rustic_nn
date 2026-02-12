import numpy as np

from raw.c5.conv_2d import _pad_2d_obs
from raw.c5.helper import assert_dim
type ndarray = np.ndarray

def _pad_2d_channel(inp: ndarray,
                    num: int):
    '''
    inp has dimension [num_channels, image_width, image_height]
    '''
    return np.stack([_pad_2d_obs(channel, num) for channel in inp])

def _pad_conv_input(inp: ndarray,
                    num: int):
    '''
    inp has dimension [batch_size, num_channels, image_width, image_height]
    '''
    return np.stack([_pad_2d_channel(obs, num) for obs in inp])

def _compute_output_obs(obs: ndarray, param: ndarray):
  '''
  obs: [channels, img_width, img_height] img_width == img_height
  param: [in_channels, out_channels, pw, ph] pw == ph
  '''
  assert_dim(obs, 3)
  assert_dim(param, 4)
  in_channels, out_channels, param_size, ph = param.shape
  param_mid = param_size // 2
  obs_pad = _pad_2d_channel(obs, param_mid)
  img_size = obs.shape[1]

  out = np.zeros((out_channels,) + obs.shape[1:])

  for c_in in range(in_channels):
     for c_out in range(out_channels):
        for o_w in range(img_size):
           for o_h in range(img_size):
              for p_w in range(param_size):
                 for p_h in range(param_size):
                    out[c_out][o_w][o_h] += param[c_in][c_out][p_w][p_h] * obs_pad[c_in][o_w + p_w][o_h + p_h]

  return out

def _output(inp: ndarray, param: ndarray) -> ndarray:
  '''
  obs [batch_size, chans, img_w, img_h] img_w == img_h
  param[in_chan, out_chan, pw, ph] pw == ph
  '''
  return np.stack([_compute_output_obs(obs, param) for obs in inp])

def _compute_output_sum(imgs: ndarray,
                        param: ndarray):
    return _output(imgs, param).sum()

# cifar_imgs = np.random.randn(10, 3, 32, 32)
# cifar_param = np.random.randn(3, 16, 5, 5)

# out = _compute_output_obs(cifar_imgs[0], cifar_param)
# assert np.allclose(out.shape,(16,32,32))

# out = _output(cifar_imgs, cifar_param)
# assert np.allclose(out.shape, [10,16,32,32])

def _compute_grads_obs(input_obs: ndarray,
                       output_grad_obs: ndarray,
                       param: ndarray) -> ndarray:
    '''
    input_obs: [in_channels, img_width, img_height]
    output_grad_obs: [out_channels, img_width, img_height]
    param: [in_channels, out_channels, img_width, img_height]
    '''
    input_grad = np.zeros_like(input_obs)
    param_size = param.shape[2]
    param_mid = param_size // 2
    img_size = input_obs.shape[1]
    in_channels = input_obs.shape[0]
    out_channels = param.shape[1]
    output_obs_pad = _pad_2d_channel(output_grad_obs, param_mid)

    for c_in in range(in_channels):
        for c_out in range(out_channels):
            for i_w in range(input_obs.shape[1]):
                for i_h in range(input_obs.shape[2]):
                    for p_w in range(param_size):
                        for p_h in range(param_size):
                            input_grad[c_in][i_w][i_h] += \
                            output_obs_pad[c_out][i_w+param_size-p_w-1][i_h+param_size-p_h-1] \
                            * param[c_in][c_out][p_w][p_h]
    return input_grad

def _input_grad(inp: ndarray,
                output_grad: ndarray,
                param: ndarray) -> ndarray:

    grads = [_compute_grads_obs(inp[i], output_grad[i], param) for i in range(output_grad.shape[0])]

    return np.stack(grads)

def _param_grad(inp: ndarray,
                output_grad: ndarray,
                param: ndarray) -> ndarray:
    '''
    inp: [in_channels, img_width, img_height]
    output_grad_obs: [out_channels, img_width, img_height]
    param: [in_channels, out_channels, img_width, img_height]
    '''
    param_grad = np.zeros_like(param)
    param_size = param.shape[2]
    param_mid = param_size // 2
    img_size = inp.shape[2]
    in_channels = inp.shape[1]
    out_channels = output_grad.shape[1]

    inp_pad = _pad_conv_input(inp, param_mid)
    img_shape = output_grad.shape[2:]

    for i in range(inp.shape[0]):
        for c_in in range(in_channels):
            for c_out in range(out_channels):
                for o_w in range(img_shape[0]):
                    for o_h in range(img_shape[1]):
                        for p_w in range(param_size):
                            for p_h in range(param_size):
                                param_grad[c_in][c_out][p_w][p_h] += \
                                inp_pad[i][c_in][o_w+p_w][o_h+p_h] \
                                * output_grad[i][c_out][o_w][o_h]
    return param_grad