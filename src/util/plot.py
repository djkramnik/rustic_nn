import numpy as np
import matplotlib.pyplot as plt
from util.customtypes import *

def plot_regression_diagnostics(y_true: ndarray, y_pred:  ndarray, model_name: str=""):
  '''
  Quick visual check of regression performance
  shows
    1. y_true vs y_pred with y=x line
    2. residuals vs y_pred
    3. histogram of residuals
  '''
  y_true_flat = y_true.reshape(-1)
  y_pred_flat = y_pred.reshape(-1)
  residuals = y_true_flat - y_pred_flat
  fig, axes = plt.subplots(1, 3, figsize=(15,4))

  # 1) subplot 1 y_true vs y_pred
  ax = axes[0]
  ax.scatter(y_true_flat, y_pred_flat, alpha=0.7)
  # create a scatter plot with points (x, y) where the x value is the target value and y value is the prediction
  # if this were perfect all points would lie on the illustrative diagonal line y=x (plotted below)
  ax.set_xlabel("True value")
  ax.set_ylabel("Predicted value")
  ax.set_title(f"{model_name}:\nTrue vs Predicted")

  min_val = float(min(y_true_flat.min(), y_pred_flat.min())) # the smallest value across both preds and true
  max_val = float(max(y_true_flat.max(), y_pred_flat.max())) # the largest value across both preds and true

  ax.plot([min_val, max_val], [min_val, max_val], linestyle="--") # draw a line between (min, min) and (max, max)

  # 2) residuals vs prediction
  ax = axes[1]
  ax.scatter(y_pred_flat, residuals, alpha=0.7)
  # create a scatter plot with points where each point (x, y) is the (prediction, error) respectively
  ax.axhline(0.0, linestyle="--")
  ax.set_xlabel("Predicted value")
  ax.set_ylabel("Residual (true - pred)")
  ax.set_title(f"{model_name}:")

  # 3) residuals histogram
  ax = axes[2]
  ax.hist(residuals, bins=20)
  ax.axvline(0.0, linestyle="--")
  ax.set_xlabel("Residual")
  ax.set_ylabel("Count")
  ax.set_title(f"{model_name}:\nResidual distribution")


  fig.suptitle(f"Regression diagnostics: {model_name}")
  fig.tight_layout()
  plt.show()

