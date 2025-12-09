import numpy as np
from optimizer.sgd import SGD
from neural_network.nn import NeuralNetwork
from loss.loss import Loss

def test_sgd_step():
  class MockLayer:
    def __init__(self, params, param_grads):
      self.params = params
      self.param_grads = param_grads

  l0p = np.array([[1., 2., 3., 4., 5.]])
  l0pg = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
  layer0 = MockLayer(l0p, l0pg)

  l1p = np.array([[2., 3., 4., 5., 6.]])
  l1pg = np.array([[0.2, 0.3, 0.4, 0.5, 0.6]])
  layer1 = MockLayer(l1p, l1pg)

  # loss irrelevant for this test
  nn = NeuralNetwork(layers = [layer0, layer1], loss = Loss())

  sgd = SGD(lr = 1.)

  # orchestrator needs to set this on the sgd instance
  sgd.__setattr__('net', nn)

  # doing this updates the params within each layer
  sgd.step()

  assert(np.allclose(layer0.params, np.array([[0.9, 1.8, 2.7, 3.6, 4.5]])))
  assert(np.allclose(layer1.params, np.array([[1.8, 2.7, 3.6, 4.5, 5.4]])))





