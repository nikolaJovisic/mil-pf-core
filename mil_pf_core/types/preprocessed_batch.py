from dataclasses import dataclass
import torch


@dataclass
class PreprocessedBatch:
    images: torch.Tensor          # (batch, height, width)
    instance_types: torch.Tensor  # (batch,)
    group_indices: torch.Tensor   # (batch,)

    def __post_init__(self):
        if not isinstance(self.images, torch.Tensor):
            raise TypeError("images must be a torch.Tensor.")
        if not isinstance(self.instance_types, torch.Tensor):
            raise TypeError("instance_types must be a torch.Tensor.")
        if not isinstance(self.group_indices, torch.Tensor):
            raise TypeError("group_indices must be a torch.Tensor.")

        if self.images.ndim != 3:
            raise ValueError("images must have shape (batch, height, width).")
        if self.instance_types.ndim != 1:
            raise ValueError("instance_types must have shape (batch,).")
        if self.group_indices.ndim != 1:
            raise ValueError("group_indices must have shape (batch,).")

        batch_size = self.images.shape[0]

        if self.instance_types.shape[0] != batch_size:
            raise ValueError("instance_types batch dimension must match images.")
        if self.group_indices.shape[0] != batch_size:
            raise ValueError("group_indices batch dimension must match images.")
