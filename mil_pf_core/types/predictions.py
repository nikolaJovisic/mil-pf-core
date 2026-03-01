from dataclasses import dataclass
import numpy as np


@dataclass
class Predictions:
    suspicious: np.ndarray   # (batch,)
    heatmap: np.ndarray      # (batch, height, width)

    def __post_init__(self):
        if not isinstance(self.suspicious, np.ndarray):
            raise TypeError("suspicious must be a numpy.ndarray.")

        if self.suspicious.ndim != 1:
            raise ValueError("suspicious must have shape (batch,).")

        if self.suspicious.dtype != np.bool_:
            raise ValueError("suspicious must be a boolean array.")

        if not isinstance(self.heatmap, np.ndarray):
            raise TypeError("heatmap must be a numpy.ndarray.")

        if self.heatmap.ndim != 3:
            raise ValueError(
                "heatmap must have shape (batch, height, width)."
            )

        if self.heatmap.shape[0] != self.suspicious.shape[0]:
            raise ValueError(
                "Batch dimension of heatmap must match suspicious."
            )
