from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class PreprocessingConfig:
    aspect_ratio: float = 1
    resize: Optional[Tuple[int, int]] = None  # (height, width)
    pad_to_max_shape: bool = True

    apply_otsu_crop: bool = True
    apply_flip: bool = True
    apply_negate: bool = True
    breast_mask_dilation_factor: int = 10

    scale_to_unit_interval: bool = True

    def __post_init__(self):
        if self.aspect_ratio <= 0:
            raise ValueError("aspect_ratio must be > 0.")
        if self.resize is not None:
            if len(self.resize) != 2:
                raise ValueError("resize must be (height, width).")
            if self.resize[0] <= 0 or self.resize[1] <= 0:
                raise ValueError("resize dimensions must be > 0.")
        if self.breast_mask_dilation_factor <= 0:
            raise ValueError("breast_mask_dilation_factor must be > 0.")
