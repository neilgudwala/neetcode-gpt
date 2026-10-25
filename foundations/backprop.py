
import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(
        self,
        x: NDArray[np.float64],
        w: NDArray[np.float64],
        b: float,
        y_true: float
    ) -> Tuple[NDArray[np.float64], float]:

        # Forward pass
        z = np.dot(x, w) + b
        y_hat = 1 / (1 + np.exp(-z))

        # Derivative of loss with respect to prediction
        dL_dy = y_hat - y_true

        # Derivative of sigmoid
        dy_dz = y_hat * (1 - y_hat)

        # Chain rule
        dL_dz = dL_dy * dy_dz

        # Gradients
        dL_dw = dL_dz * x
        dL_db = dL_dz

        # Round to 5 decimal places
        dL_dw = np.round(dL_dw, 5)
        dL_db = round(float(dL_db), 5)

        return dL_dw, dL_db
