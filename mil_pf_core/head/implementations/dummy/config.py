from dataclasses import dataclass
from typing import Tuple

from mil_pf_core.head.config import HeadConfig


@dataclass
class DummyHeadConfig(HeadConfig):
    heatmap_shape: Tuple[int, int] = (8, 8)

    def __post_init__(self):
        super().__post_init__()
        if self.heatmap_shape is None:
            raise ValueError("heatmap_shape must not be None.")
        if len(self.heatmap_shape) != 2:
            raise ValueError("heatmap_shape must be (height, width).")
        if self.heatmap_shape[0] <= 0 or self.heatmap_shape[1] <= 0:
            raise ValueError("heatmap_shape dimensions must be > 0.")
