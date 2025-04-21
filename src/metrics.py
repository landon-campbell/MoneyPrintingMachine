import numpy as np

def mse(real: np.ndarray, pred: np.ndarray) -> float:
    return np.mean((real - pred)**2)
