from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class ViewSet:
    images: List[np.ndarray]

    def __post_init__(self):
        for img in self.images:
            if not isinstance(img, np.ndarray):
                raise TypeError("All images must be numpy arrays.")
            if img.dtype != np.uint16:
                raise ValueError("All images must be 16-bit arrays.")
            if img.ndim != 2:
                raise ValueError("All images must be 2D arrays (ndim == 2).")
