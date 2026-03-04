from typing import Union

from mil_pf_core.embedder.implementations.dinov2 import DinoV2Embedder
from mil_pf_core.embedder.implementations.dinov2 import DinoV2EmbedderConfig
from mil_pf_core.embedder.implementations.dummy import DummyEmbedder
from mil_pf_core.embedder.implementations.dummy import DummyEmbedderConfig
from mil_pf_core.embedder.implementations.medsiglip import MedSigLIPEmbedder
from mil_pf_core.embedder.implementations.medsiglip import MedSigLIPEmbedderConfig
from mil_pf_core.embedder.interface import EmbedderInterface


EmbedderBuildConfig = Union[
    DummyEmbedderConfig, DinoV2EmbedderConfig, MedSigLIPEmbedderConfig
]


def create_embedder(config: EmbedderBuildConfig) -> EmbedderInterface:
    if isinstance(config, DummyEmbedderConfig):
        return DummyEmbedder(config)
    if isinstance(config, DinoV2EmbedderConfig):
        return DinoV2Embedder(config)
    if isinstance(config, MedSigLIPEmbedderConfig):
        return MedSigLIPEmbedder(config)
    raise TypeError(f"Unsupported embedder config type: {type(config).__name__}")
