"""Module that compiles loss functions.

Module that stores common loss functions used throughout the repository.
- PHMAP Data Challenge Loss Function
"""
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from typing import Tuple


def phmap_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the PHMAP 2021 Data Challenge Loss.

    Parameters
    ----------
    y_true : np.ndarray
        True labels for RUL
    y_pred : np.ndarray
        Predicted labels for RUL

    Returns
    -------
    float
        Average of RMSE and NASA's scoring function

    - Check the provided paper for further documentation
    """
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    diff = y_true - y_pred
    alpha = np.where(diff <= 0, -1/10, 1/13)
    nasa = np.mean(np.exp(alpha * diff) - 1)

    return 0.5*(rmse + nasa)


def PHMAP(y_pred: np.ndarray, y_true: xgb.DMatrix) -> Tuple[str, float]:
    """Calculate PHMAP loss for XGBoost regressor.

    Parameters
    ----------
    y_pred : np.ndarray
        Predictions from trained XGB model
    y_true : xgb.DMatrix
        Real RUL labels

    Returns
    -------
    Tuple[str, float]
        Loss name and value
    """
    y = y_true.get_label()

    return 'PHMAP', phmap_loss(y_true=y, y_pred=y_pred)
