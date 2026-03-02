import torch

from mil_pf_core.embedding.interface import EmbeddingInterface
from mil_pf_core.types.embeddings import Embeddings
from mil_pf_core.types.images import Images


class DummyEmbedding(EmbeddingInterface):
    def __init__(self, embedding_dim: int = 16):
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be > 0.")
        self.embedding_dim = embedding_dim

    def embed(self, images: Images) -> Embeddings:
        batch_size = images.images.shape[0]
        return Embeddings(embeddings=torch.rand(batch_size, self.embedding_dim))
