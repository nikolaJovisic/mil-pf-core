from typing import Union

from mil_pf_core.head.implementations.dummy import DummyHead
from mil_pf_core.head.implementations.dummy import DummyHeadConfig
from mil_pf_core.head.implementations.mil_pf_attn import MILPFAttnHead
from mil_pf_core.head.implementations.mil_pf_attn import MILPFAttnHeadConfig
from mil_pf_core.head.interface import HeadInterface


HeadBuildConfig = Union[DummyHeadConfig, MILPFAttnHeadConfig]


def create_head(config: HeadBuildConfig) -> HeadInterface:
    if isinstance(config, DummyHeadConfig):
        return DummyHead(config)
    if isinstance(config, MILPFAttnHeadConfig):
        return MILPFAttnHead(config)
    raise TypeError(f"Unsupported head config type: {type(config).__name__}")
