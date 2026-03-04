from dataclasses import dataclass
from typing import Tuple

from mil_pf_core.embedder.config import EmbedderConfig


@dataclass
class MedSigLIPEmbedderConfig(EmbedderConfig):
    input_shape: Tuple[int, int] = (448, 448)
    device: str = "cpu"
    model_name: str = "google/medsiglip-448"

    def __post_init__(self):
        super().__post_init__()
        if self.device is None or not str(self.device).strip():
            raise ValueError("device must not be None or empty.")
        if self.model_name is None or not str(self.model_name).strip():
            raise ValueError("model_name must not be None or empty.")
