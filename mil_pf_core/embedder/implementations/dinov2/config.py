from dataclasses import dataclass

from mil_pf_core.embedder.config import EmbedderConfig


@dataclass
class DinoV2EmbedderConfig(EmbedderConfig):
    device: str = "cpu"
    repo: str = "facebookresearch/dinov2"
    model_name: str = "dinov2_vitg14"

    def __post_init__(self):
        super().__post_init__()
        if self.device is None or not str(self.device).strip():
            raise ValueError("device must not be None or empty.")
        if self.repo is None or not str(self.repo).strip():
            raise ValueError("repo must not be None or empty.")
        if self.model_name is None or not str(self.model_name).strip():
            raise ValueError("model_name must not be None or empty.")
