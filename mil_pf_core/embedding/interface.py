from abc import ABC, abstractmethod

from mil_pf_core.types.embeddings import Embeddings
from mil_pf_core.types.images import Images


class EmbeddingInterface(ABC):
    @abstractmethod
    def embed(self, images: Images) -> Embeddings:
        """Maps Images into Embeddings."""
