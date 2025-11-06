import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_boston():
  current_dir = os.path.dirname(os.path.abspath(__file__))
  boston = pd.read_csv(os.path.join(current_dir, 'boston.txt'), sep="\s+", skiprows=22, header=None)
  data = np.hstack([boston.values[::2, :], boston.values[1::2, :2]])
  target = boston.values[1::2, 2]
  features = [
    'CRIM',
    'ZN',
    'INDUS',
    'CHAS',
    'NOX',
    'RM',
    'AGE',
    'DIS',
    'RAD',
    'TAX',
    'PTRATIO',
    'B',
    'LSTAT',
    'MEDV'
  ]
  return [data, target, features]

def bootstrap_boston(debug=False):
  [data, target, features] = load_boston()
  if (debug):
    print("loading boston...")
    print("x: ", data[0])
    print("y: ", target[0])
    print(features)

  # normalize data
  s = StandardScaler()
  data = s.fit_transform(data)
  if (debug):
    print("normalized boston data: ", data[0])
  X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.3, random_state=80718)
  y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)
  return X_train, X_test, y_train, y_test
