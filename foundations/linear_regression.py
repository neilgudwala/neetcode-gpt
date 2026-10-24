import numpy as np
from numpy.typing import NDArray


class Solution:

    def get_model_prediction(
        self,
        X: NDArray[np.float64],
        weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # Compute predictions using matrix multiplication
        predictions = np.dot(X, weights)

        # Round to 5 decimal places
        return np.round(predictions, 5)

    def get_error(
        self,
        model_prediction: NDArray[np.float64],
        ground_truth: NDArray[np.float64]
    ) -> float:
        # Compute mean squared error
        mse = np.mean((model_prediction - ground_truth) ** 2)

        # Round to 5 decimal places
        return round(float(mse), 5)