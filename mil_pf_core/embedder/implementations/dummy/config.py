from dataclasses import dataclass

from mil_pf_core.embedder.config import EmbedderConfig


@dataclass
class DummyEmbedderConfig(EmbedderConfig):
    embedding_dim: int = 16

    def __post_init__(self):
        super().__post_init__()
        if self.embedding_dim is None:
            raise ValueError("embedding_dim must not be None.")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be > 0.")
