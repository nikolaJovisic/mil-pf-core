from dataclasses import dataclass
import torch
from mil_pf_core.types.embeddings import Embeddings


@dataclass
class StructuredEmbeddings:
    embeddings: Embeddings
    instance_types: torch.Tensor  # (batch,)
    groups: torch.Tensor  # (batch,)

    def __post_init__(self):
        if not isinstance(self.embeddings, Embeddings):
            raise TypeError("embeddings must be an Embeddings.")
        if not isinstance(self.instance_types, torch.Tensor):
            raise TypeError("instance_types must be a torch.Tensor.")
        if not isinstance(self.groups, torch.Tensor):
            raise TypeError("groups must be a torch.Tensor.")

        if self.instance_types.ndim != 1:
            raise ValueError("instance_types must have shape (batch,).")
        if self.groups.ndim != 1:
            raise ValueError("groups must have shape (batch,).")

        batch_size = self.embeddings.embeddings.shape[0]

        if self.instance_types.shape[0] != batch_size:
            raise ValueError(
                "instance_types batch dimension must match embeddings."
            )
        if self.groups.shape[0] != batch_size:
            raise ValueError("groups batch dimension must match embeddings.")
