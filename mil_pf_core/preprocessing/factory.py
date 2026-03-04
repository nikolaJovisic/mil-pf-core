from typing import Union

from mil_pf_core.preprocessing.implementations.dummy import DummyPreprocessing
from mil_pf_core.preprocessing.implementations.dummy import DummyPreprocessingConfig
from mil_pf_core.preprocessing.implementations.mammo import MammoPreprocessing
from mil_pf_core.preprocessing.implementations.mammo import MammoPreprocessingConfig
from mil_pf_core.preprocessing.interface import PreprocessingInterface


PreprocessingBuildConfig = Union[
    DummyPreprocessingConfig, MammoPreprocessingConfig
]


def create_preprocessing(config: PreprocessingBuildConfig) -> PreprocessingInterface:
    if isinstance(config, DummyPreprocessingConfig):
        return DummyPreprocessing(config)
    if isinstance(config, MammoPreprocessingConfig):
        return MammoPreprocessing(config)
    raise TypeError(f"Unsupported preprocessing config type: {type(config).__name__}")
