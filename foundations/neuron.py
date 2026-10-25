import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(
        self,
        x: NDArray[np.float64],
        w: NDArray[np.float64],
        b: float,
        activation: str
    ) -> float:
        # Pre-activation
        z = np.dot(x, w) + b

        # Apply activation function
        if activation == "sigmoid":
            ans = 1 / (1 + np.exp(-z))
        elif activation == "relu":
            ans = max(0, z)
        else:
            raise ValueError("Unsupported activation function")

        return round(float(ans), 5)