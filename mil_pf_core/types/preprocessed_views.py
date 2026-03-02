from dataclasses import dataclass
import torch
from mil_pf_core.types.images import Images


@dataclass
class PreprocessedViews:
    images: Images
    instance_types: torch.Tensor  # (batch,)
    groups: torch.Tensor   # (batch,)

    def __post_init__(self):
        if not isinstance(self.images, Images):
            raise TypeError("images must be an Images.")
        if not isinstance(self.instance_types, torch.Tensor):
            raise TypeError("instance_types must be a torch.Tensor.")
        if not isinstance(self.groups, torch.Tensor):
            raise TypeError("groups must be a torch.Tensor.")

        if self.instance_types.ndim != 1:
            raise ValueError("instance_types must have shape (batch,).")
        if self.groups.ndim != 1:
            raise ValueError("groups must have shape (batch,).")
        if torch.is_floating_point(self.instance_types):
            raise ValueError("instance_types must be an integer tensor.")
        if not torch.all(
            (self.instance_types == 0) | (self.instance_types == 1)
        ):
            raise ValueError("instance_types must contain only 0 and 1 values.")

        batch_size = self.images.images.shape[0]

        if self.instance_types.shape[0] != batch_size:
            raise ValueError("instance_types batch dimension must match images.")
        if self.groups.shape[0] != batch_size:
            raise ValueError("groups batch dimension must match images.")
