from dataclasses import dataclass

from mil_pf_core.preprocessing.config import PreprocessingConfig


@dataclass
class DummyPreprocessingConfig(PreprocessingConfig):
    output_shape: tuple[int, int] = (32, 32)
    tiles_per_image: int = 6

    def __post_init__(self):
        super().__post_init__()
        if self.tiles_per_image is None:
            raise ValueError("tiles_per_image must not be None.")
        if self.tiles_per_image < 0:
            raise ValueError("tiles_per_image must be >= 0.")
