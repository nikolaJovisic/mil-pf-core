from dataclasses import dataclass
import torch


@dataclass
class Images:
    images: torch.Tensor  # (batch, height, width)

    def __post_init__(self):
        if not isinstance(self.images, torch.Tensor):
            raise TypeError("images must be a torch.Tensor.")

        if self.images.ndim != 3:
            raise ValueError("images must have shape (batch, height, width).")
