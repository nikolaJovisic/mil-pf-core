from dataclasses import dataclass
from typing import Tuple


@dataclass
class EmbedderConfig:
    input_shape: Tuple[int, int] = (518, 518)

    def __post_init__(self):
        if self.input_shape is None:
            raise ValueError("input_shape must not be None.")
        if len(self.input_shape) != 2 or self.input_shape[0] <= 0 or self.input_shape[1] <= 0:
            raise ValueError("input_shape must be (height, width) with positive values.")
