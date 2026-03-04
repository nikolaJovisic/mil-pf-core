from typing import Union

from mil_pf_core.head.implementations.dummy import DummyHead
from mil_pf_core.head.implementations.dummy import DummyHeadConfig
from mil_pf_core.head.implementations.milpf import MILPFHead
from mil_pf_core.head.implementations.milpf import MILPFHeadConfig
from mil_pf_core.head.interface import HeadInterface


HeadBuildConfig = Union[DummyHeadConfig, MILPFHeadConfig]


def create_head(config: HeadBuildConfig) -> HeadInterface:
    if isinstance(config, DummyHeadConfig):
        return DummyHead(config)
    if isinstance(config, MILPFHeadConfig):
        return MILPFHead(config)
    raise TypeError(f"Unsupported head config type: {type(config).__name__}")
