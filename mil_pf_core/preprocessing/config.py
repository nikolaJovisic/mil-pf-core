from dataclasses import dataclass
from typing import Tuple


@dataclass
class PreprocessingConfig:
    output_shape: Tuple[int, int] = (518, 518)

    def __post_init__(self):
        if self.output_shape is None:
            raise ValueError("output_shape must not be None.")
        if len(self.output_shape) != 2:
            raise ValueError("output_shape must be (height, width).")
        if self.output_shape[0] <= 0 or self.output_shape[1] <= 0:
            raise ValueError("output_shape dimensions must be > 0.")
