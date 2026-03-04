from abc import ABC, abstractmethod
from typing import Tuple

from mil_pf_core.types.embeddings import Embeddings
from mil_pf_core.types.images import Images


class EmbedderInterface(ABC):
    @property
    @abstractmethod
    def input_shape(self) -> Tuple[int, int]:
        """Expected (height, width) for embedder inputs."""

    @abstractmethod
    def embed(self, images: Images) -> Embeddings:
        """Maps Images into Embeddings."""
