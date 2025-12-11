import numpy as np
from copy import deepcopy
from typing import Iterator, Callable, Optional

from neural_network.nn import NeuralNetwork
from optimizer.optimizer import Optimizer
from util.customtypes import *

# shuffle X; same shuffle for y
def permute_data(X: ndarray, y: ndarray) -> tuple[ndarray, ndarray]:
  perm = np.random.permutation(X.shape[0])
  return X[perm], y[perm]

class Trainer(object):
  def __init__(
    self,
    net: NeuralNetwork,
    optim: Optimizer,
    eval_callback: Optional[Callable[[float, float, int], None]] = None,
  ):
    '''
    Requires a neural network and an optimizer.
    Optionally accepts eval_callback(train_loss, val_loss, epoch_idx)
    which is invoked during fit() at each evaluation step.
    '''
    self.net = net
    self.optim = optim
    setattr(self.optim, 'net', self.net)
    self.best_loss = 1e9
    self.eval_callback = eval_callback

  def generate_batches(self, X: ndarray, y: ndarray, size: int = 32) -> Iterator[tuple[ndarray, ndarray]]:
    '''
    Given the complete training dataset, split into batches of size `size`.
    '''
    N = X.shape[0]
    assert N == y.shape[0]
    for i in range(0, N, size):
      X_batch = X[i:i+size]
      y_batch = y[i:i+size]
      yield X_batch, y_batch

  # do everything
  def fit(
    self,
    X_train: ndarray,
    y_train: ndarray,
    X_test: ndarray,
    y_test: ndarray,
    epochs: int = 100,
    eval_every: int = 10,
    batch_size: int = 32,
    seed: int = 1,
    restart: bool = True
  ) -> None:
    '''
    Fits the neural network on the training data for a certain number
    of epochs. Run validation every `eval_every` iterations.
    Optionally calls eval_callback(train_loss, val_loss, epoch_idx).
    '''
    np.random.seed(seed)

    # this will cause each layer to invoke setup_layer.
    # in the case of Dense this reinstantiates the ParamOperations with fresh params
    if restart:
      for layer in self.net.layers:
        layer.first = True
      self.best_loss = 1e9

    best_model = deepcopy(self.net)

    for e in range(epochs):
      # before chunking, shuffle
      X_train, y_train = permute_data(X_train, y_train)
      batch_generator = self.generate_batches(
        X_train, y_train, batch_size
      )

      for (X_batch, y_batch) in batch_generator:
        self.net.train_batch(X_batch, y_batch)
        self.optim.step()

      # separate from training. eval for log and early stop
      if (e + 1) % eval_every == 0:
        # full-batch train + val evaluation
        train_preds = self.net.forward(X_train)
        train_loss = self.net.loss.forward(train_preds, y_train)

        test_preds = self.net.forward(X_test)
        val_loss = self.net.loss.forward(test_preds, y_test)

        # optional user callback
        if self.eval_callback is not None:
          # pass train_loss, val_loss, and epoch index (1-based)
          self.eval_callback(train_loss, val_loss, e + 1)

        if val_loss < self.best_loss:
          print(f"Validation loss after {e+1} epochs is {val_loss:.3f}")
          self.best_loss = val_loss
          best_model = deepcopy(self.net)
        else:
          break

    self.net = best_model
    setattr(self.optim, 'net', self.net)
