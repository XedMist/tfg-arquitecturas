from dataclasses import dataclass, field
from typing import Any, Dict, List, Type

import torch
import torch.nn as nn


@dataclass
class BaseMixerConfig:
    d_model: int
    dropout: float = 0.0
