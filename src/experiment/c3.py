import numpy as np
from util.customtypes import *
from data.boston import bootstrap_boston
from trainer.trainer import Trainer
from layer.dense import Dense
from operation.activation import Linear, Sigmoid
from neural_network.nn import NeuralNetwork
from loss.mse import MeanSquaredError
from optimizer.sgd import SGD

boston_data = bootstrap_boston()

X_train, y_train, X_test, y_test = boston_data["data"] # ndarray
features = boston_data["features"] # list[str]

# sacred seed for computation ritual
seed = 20190501

# three kinds of neural networks

# control? AKA linear regression?
lr = NeuralNetwork(
  layers=[
    Dense(neurons=1, activation=Linear())
  ],
  loss = MeanSquaredError(),
  seed=seed
)

nn = NeuralNetwork(
  layers = [
    Dense(neurons=13, activation=Sigmoid()),
    Dense(neurons=1, activation=Linear()),
  ],
  loss = MeanSquaredError(),
  seed=seed
)

dl = NeuralNetwork(
  layers = [
    Dense(neurons=13, activation=Sigmoid()),
    Dense(neurons=13, activation = Sigmoid()),
    Dense(neurons=1, activation=Linear())
  ],
  loss = MeanSquaredError(),
  seed=seed
)

# evaluation
def mae(y_true: ndarray, y_pred: ndarray):
    '''
    Compute mean absolute error for a neural network.
    '''
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true: ndarray, y_pred: ndarray):
    '''
    Compute root mean squared error for a neural network.
    '''
    return np.sqrt(np.mean(np.power(y_true - y_pred, 2)))

def eval_regression_model(model: NeuralNetwork,
                          X_test: ndarray,
                          y_test: ndarray):
    '''
    Compute mae and rmse for a neural network.
    '''
    preds = model.forward(X_test)
    preds = preds.reshape(-1, 1)
    print("Mean absolute error: {:.2f}".format(mae(preds, y_test)))
    print()
    print("Root mean squared error {:.2f}".format(rmse(preds, y_test)))

trainer = Trainer(lr, SGD(lr=0.01))

trainer.fit(X_train, y_train, X_test, y_test,
       epochs = 50,
       eval_every = 10,
       seed=seed)
print()
eval_regression_model(lr, X_test, y_test)







