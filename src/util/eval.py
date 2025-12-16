import numpy as np
from util.customtypes import ndarray

from neural_network.nn import NeuralNetwork

def calc_accuracy_model(model: NeuralNetwork, test: ndarray, target: ndarray):
  return print(
    '''The model validation accuracy is: {0:.2f}%'''.format(
      np.equal(np.argmax(model.forward(test), axis=1), target).sum()
      * 100.0
      / test.shape[0]
    )
  )