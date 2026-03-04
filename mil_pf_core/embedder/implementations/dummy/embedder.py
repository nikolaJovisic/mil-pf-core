from typing import Tuple

import torch

from mil_pf_core.embedder.implementations.dummy.config import DummyEmbedderConfig
from mil_pf_core.embedder.interface import EmbedderInterface
from mil_pf_core.types.embeddings import Embeddings
from mil_pf_core.types.images import Images


class DummyEmbedder(EmbedderInterface):
    def __init__(self, config: DummyEmbedderConfig):
        self.config = config

    @property
    def input_shape(self) -> Tuple[int, int]:
        return self.config.input_shape

    def embed(self, images: Images) -> Embeddings:
        batch_size = images.images.shape[0]
        return Embeddings(embeddings=torch.rand(batch_size, self.config.embedding_dim))
