import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    It returns the accuracy of the model on the given dataset

    Parameters
    ----------
    y_true: np.ndarray
        The true labels of the dataset
    y_pred: np.ndarray
        The predicted labels of the dataset

    Returns
    -------
    accuracy: float
        The accuracy of the model
    """
    # deal with predictions like [[0.52], [0.91], ...] and [[0.3, 0.7], [0.6, 0.4], ...]
    # they need to be in the same format: [0, 1, ...] and [1, 0, ...]
    return np.sum(y_true == y_pred) / len(y_true)