from dataclasses import dataclass
from typing import Tuple

from mil_pf_core.preprocessing.config import PreprocessingConfig


@dataclass
class MammoPreprocessingConfig(PreprocessingConfig):
    aspect_ratio: float = 1.0
    shape: Tuple[int, int] = (4096, 4096)
    breast_mask_dilation_factor: int = 10
    tile_overlap: float = 0.25
    tile_threshold: float = 0.05
    tile_increase_tolerance: float = 0.1

    def __post_init__(self):
        super().__post_init__()
        if self.aspect_ratio is None:
            raise ValueError("aspect_ratio must not be None.")
        if self.aspect_ratio <= 0:
            raise ValueError("aspect_ratio must be > 0.")
        if self.shape is None:
            raise ValueError("shape must not be None.")
        if len(self.shape) != 2:
            raise ValueError("shape must be (height, width).")
        if self.shape[0] <= 0 or self.shape[1] <= 0:
            raise ValueError("shape dimensions must be > 0.")
        if self.breast_mask_dilation_factor is None:
            raise ValueError("breast_mask_dilation_factor must not be None.")
        if self.breast_mask_dilation_factor <= 0:
            raise ValueError("breast_mask_dilation_factor must be > 0.")
        if self.tile_overlap is None:
            raise ValueError("tile_overlap must not be None.")
        if not 0 <= self.tile_overlap < 1:
            raise ValueError("tile_overlap must satisfy 0 <= tile_overlap < 1.")
        if self.tile_threshold is None:
            raise ValueError("tile_threshold must not be None.")
        if self.tile_threshold < 0:
            raise ValueError("tile_threshold must be >= 0.")
        if self.tile_increase_tolerance is None:
            raise ValueError("tile_increase_tolerance must not be None.")
        if self.tile_increase_tolerance < 0:
            raise ValueError("tile_increase_tolerance must be >= 0.")
