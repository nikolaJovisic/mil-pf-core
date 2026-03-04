from typing import Tuple

import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from mil_pf_core.embedder.implementations.dinov2.config import DinoV2EmbedderConfig
from mil_pf_core.embedder.interface import EmbedderInterface
from mil_pf_core.types.embeddings import Embeddings
from mil_pf_core.types.images import Images


class DinoV2Embedder(EmbedderInterface):
    def __init__(self, config: DinoV2EmbedderConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.model = torch.hub.load(config.repo, config.model_name).to(self.device).eval()

    @property
    def input_shape(self) -> Tuple[int, int]:
        return self.config.input_shape

    def embed(self, images: Images) -> Embeddings:
        tensor = images.images
        if tensor.ndim != 3:
            raise ValueError("Expected images tensor with shape (batch, height, width).")
        if tuple(tensor.shape[1:]) != self.config.input_shape:
            raise ValueError(
                f"DinoV2 expected input shape {self.config.input_shape}, got {tuple(tensor.shape[1:])}."
            )

        x = tensor.to(dtype=torch.float32)
        x = repeat(x, "b h w -> b c h w", c=3).to(self.device)

        with torch.inference_mode():
            outputs = self.model(x)

        if outputs.ndim != 2:
            outputs = rearrange(outputs, "b ... -> b (...)")
        outputs = F.normalize(outputs, p=2, dim=1)

        return Embeddings(embeddings=outputs.detach().cpu())
