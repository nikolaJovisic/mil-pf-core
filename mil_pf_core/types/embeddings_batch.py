from dataclasses import dataclass
import torch


@dataclass
class EmbeddingsBatch:
    embeddings: torch.Tensor  # (batch, embedding_dim)

    def __post_init__(self):
        if not isinstance(self.embeddings, torch.Tensor):
            raise TypeError("embeddings must be a torch.Tensor.")

        if self.embeddings.ndim != 2:
            raise ValueError(
                "embeddings must have shape (batch, embedding_dim)."
            )
