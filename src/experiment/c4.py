import numpy as np
from data.load_mnist import load
from util.eval import calc_accuracy_model
from util.one_hot_encode import one_hot_encode
from util.scale import scale_train_data
from neural_network.nn import NeuralNetwork
from layer.dense import Dense
from operation.activation import Sigmoid, Linear
from loss.mse import MeanSquaredError
from loss.softmax_cross_entropy import SoftmaxCrossEntropy
from trainer.trainer import Trainer
from optimizer.sgd import SGD

X_train, y_train, X_test, y_test = load()

num_mnist_labels = 10

# one hot encode targets
y_train_encoded = one_hot_encode(y_train, num_mnist_labels)
y_test_encoded = one_hot_encode(y_test, num_mnist_labels)

# scale train data (mean = 0, variance = 1)
X_train, X_test = scale_train_data(X_train, X_test)

sacred_seed = 190119

model = NeuralNetwork(
  layers=[
    Dense(neurons=89, activation=Sigmoid()),
    Dense(neurons=10, activation=Sigmoid())
  ],
  loss = MeanSquaredError(),
  seed=sacred_seed
)

trainer = Trainer(model, SGD(0.1))
trainer.fit(
  X_train,
  y_train_encoded,
  X_test,
  y_test_encoded,
  epochs=50,
  eval_every=5,
  seed=sacred_seed, # this is redundant I think either here or there
  batch_size=60,
)

calc_accuracy_model(model, X_test, y_test)

# now try with softmax cross entropy loss

model = NeuralNetwork(
  layers=[
    Dense(neurons=89, activation=Sigmoid()),
    Dense(neurons=10, activation=Linear())
  ],
  loss = SoftmaxCrossEntropy(),
  seed=sacred_seed
)

trainer = Trainer(model, SGD(0.1))
trainer.fit(
  X_train,
  y_train_encoded,
  X_test,
  y_test_encoded,
  epochs=50,
  eval_every=5,
  seed=sacred_seed, # this is redundant I think either here or there
  batch_size=60,
)

calc_accuracy_model(model, X_test, y_test)
