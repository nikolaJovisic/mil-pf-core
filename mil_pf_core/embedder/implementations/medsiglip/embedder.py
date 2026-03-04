from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from transformers import AutoModel, AutoProcessor

from mil_pf_core.embedder.implementations.medsiglip.config import MedSigLIPEmbedderConfig
from mil_pf_core.embedder.interface import EmbedderInterface
from mil_pf_core.types.embeddings import Embeddings
from mil_pf_core.types.images import Images


class MedSigLIPEmbedder(EmbedderInterface):
    def __init__(self, config: MedSigLIPEmbedderConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.processor = AutoProcessor.from_pretrained(config.model_name)
        self.model = AutoModel.from_pretrained(config.model_name).to(self.device).eval()

    @property
    def input_shape(self) -> Tuple[int, int]:
        return self.config.input_shape

    def embed(self, images: Images) -> Embeddings:
        tensor = images.images
        if tensor.ndim != 3:
            raise ValueError("Expected images tensor with shape (batch, height, width).")
        if tuple(tensor.shape[1:]) != self.config.input_shape:
            raise ValueError(
                f"MedSigLIP expected input shape {self.config.input_shape}, got {tuple(tensor.shape[1:])}."
            )

        x = tensor.to(dtype=torch.float32).clamp(0.0, 1.0)
        x = repeat(x, "b h w -> b c h w", c=3)
        x_np = (rearrange(x, "b c h w -> b h w c").cpu().numpy() * 255.0).astype(np.uint8)
        image_list = [img for img in x_np]
        inputs = self.processor(images=image_list, return_tensors="pt").to(self.device)

        with torch.inference_mode():
            pixel_values = inputs["pixel_values"]
            vision_outputs = self.model.vision_model(pixel_values=pixel_values)
            if hasattr(self.model, "visual_projection"):
                outputs = self.model.visual_projection(vision_outputs.pooler_output)
            else:
                outputs = vision_outputs.pooler_output

        if outputs.ndim != 2:
            outputs = rearrange(outputs, "b ... -> b (...)")
        outputs = F.normalize(outputs, p=2, dim=1)
        return Embeddings(embeddings=outputs.detach().cpu())
