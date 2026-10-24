import numpy as np
from numpy.typing import NDArray


class Solution:
    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # Subtract max for numerical stability
        z_stable = z - np.max(z)

        # Compute exponentials
        exp_z = np.exp(z_stable)

        # Normalize to get probabilities
        y = exp_z / np.sum(exp_z)

        return np.round(y, 4)