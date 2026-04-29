from dataclasses import dataclass, field
from typing import Any, Dict, List, Type

import torch
import torch.nn as nn


@dataclass
class BaseMixerConfig:
    d_model: int
    dropout: float = 0.0


@dataclass
class DeformableAttentionMixerConfig:
    num_heads: int = 8
    n_groups: int = 4
    stride: int = 2
    offset_range_factor: float = -1.0
    no_off: bool = False
    ksize: int = 5
    attn_drop: float = 0.0
    proj_drop: float = 0.0
    use_pe: bool = True
    dwc_pe: bool = True
    fixed_pe: bool = False
    log_cpb: bool = False
