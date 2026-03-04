from dataclasses import dataclass, field
from typing import Tuple

from mil_pf_core.head.config import HeadConfig


@dataclass
class MILPFModelConfig:
    input_dim: int = 1536
    num_classes: int = 1
    gl_hidden_dim: int = 8
    lc_hidden_dim: int = 8
    num_latents: int = 1
    mlp_out: bool = False

    def __post_init__(self):
        if self.input_dim <= 0:
            raise ValueError("input_dim must be > 0.")
        if self.num_classes <= 0:
            raise ValueError("num_classes must be > 0.")
        if self.gl_hidden_dim <= 0:
            raise ValueError("gl_hidden_dim must be > 0.")
        if self.lc_hidden_dim <= 0:
            raise ValueError("lc_hidden_dim must be > 0.")
        if self.num_latents <= 0:
            raise ValueError("num_latents must be > 0.")


@dataclass
class MILPFHeadConfig(HeadConfig):
    model: MILPFModelConfig = field(default_factory=MILPFModelConfig)
    head_path: str = ""
    strict_load: bool = False
    device: str = "cpu"
    heatmap_shape: Tuple[int, int] = (4096, 4096)

    def __post_init__(self):
        super().__post_init__()

        if self.model is None:
            raise ValueError("model config must not be None.")

        if self.head_path is None or not str(self.head_path).strip():
            raise ValueError("head_path must not be None or empty.")

        if self.device is None or not str(self.device).strip():
            raise ValueError("device must not be None or empty.")

        if self.heatmap_shape is None or len(self.heatmap_shape) != 2:
            raise ValueError("heatmap_shape must be (height, width).")
        if self.heatmap_shape[0] <= 0 or self.heatmap_shape[1] <= 0:
            raise ValueError("heatmap_shape dimensions must be > 0.")
