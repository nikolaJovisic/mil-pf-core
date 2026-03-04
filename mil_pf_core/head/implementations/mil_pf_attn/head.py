import numpy as np
import torch

from mil_pf_core.head.implementations.mil_pf_attn.config import MILPFAttnHeadConfig
from mil_pf_core.head.implementations.mil_pf_attn.model import MILPFAttnTrexModel
from mil_pf_core.head.interface import HeadInterface
from mil_pf_core.types.predictions import Predictions
from mil_pf_core.types.structured_embeddings import StructuredEmbeddings


class MILPFAttnHead(HeadInterface):
    def __init__(self, config: MILPFAttnHeadConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.model = MILPFAttnTrexModel(config.model).to(self.device).eval()

        checkpoint = torch.load(config.head_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        self.model.load_state_dict(state_dict, strict=config.strict_load)

    def predict(self, structured_embeddings: StructuredEmbeddings) -> Predictions:
        x = structured_embeddings.embeddings.embeddings.to(self.device, dtype=torch.float32)
        group = structured_embeddings.groups.to(self.device, dtype=torch.long)
        instance_type = structured_embeddings.instance_types.to(
            self.device, dtype=torch.long
        )

        with torch.inference_mode():
            logits = self.model(x, group, instance_type).squeeze(-1)
            probs = torch.sigmoid(logits)

        # Keep model outputs at case/group level.
        if probs.ndim == 0:
            probs = probs.unsqueeze(0)
        suspicious = (probs > self.config.suspicious_threshold).cpu().numpy().astype(np.bool_)
        heatmap = np.zeros(
            (probs.shape[0], self.config.heatmap_shape[0], self.config.heatmap_shape[1]),
            dtype=np.float32,
        )
        return Predictions(suspicious=suspicious, heatmap=heatmap)
