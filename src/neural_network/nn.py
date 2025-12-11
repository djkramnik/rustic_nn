from layer.layer import Layer
from loss.loss import Loss
from util.customtypes import *

class NeuralNetwork(object):
  def __init__(self, layers: list[Layer], loss: Loss, seed: int = 1):
    self.layers = layers
    self.loss = loss
    self.seed = seed
    # the benefactor... why..
    for layer in self.layers:
      setattr(layer, 'seed', self.seed)

  def forward(self, x_batch: ndarray) -> ndarray:
    x_out = x_batch
    for layer in self.layers:
      x_out = layer.forward(x_out)
    return x_out

  def backward(self, loss_grad: ndarray):
    grad = loss_grad
    for layer in reversed(self.layers):
      grad = layer.backward(grad)

  def train_batch(self, x_batch: ndarray, y_batch: ndarray) -> ndarray:
    '''
    Passes data forward through the layers,
    computes the loss,
    passes the loss_gradient back through the layers
    '''
    preds = self.forward(x_batch)
    # we don't technically need this? except perhaps
    # for logging / early stop?
    loss_value = self.loss.forward(preds, y_batch)
    # actually loss.forward was necessary if for no other
    # reason than because only calling forward sets
    # the preds and target vars :o
    self.backward(self.loss.backward())

    return loss_value

  def params(self):
    '''
    yields the params from the layers
    the layers themselves build its params from the param field where it exists in its operations
    I suppose we will be yielding these up to the trainer for adjustment
    '''
    for layer in self.layers:
      yield from layer.params
  def param_grads(self):
    '''
    yields the param_grads from the layers
    the layers themselves build its param_grads from the param_grad field where it exists in its operations
    I suppose we will be yielding these up to the trainer for adjustment
    '''
    for layer in self.layers:
      yield from layer.param_grads




