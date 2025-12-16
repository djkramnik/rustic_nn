import numpy as np
from util.customtypes import ndarray

def scale_train_data(train: ndarray, test: ndarray):
  train_mean = np.mean(train)
  train = train - train_mean
  test = test - train_mean # test uses train mean as well

  train_std = np.std(train)
  train = train / train_std
  test = test / train_std

  return train, test