from dataclasses import dataclass
import torch
import torch.nn as nn

from models.mixer.mixer_config import BaseMixerConfig
from models.mixer.registry import register_mixer


@dataclass
class PoolMixerConfig(BaseMixerConfig):
    pool_size: int = 3


@register_mixer(PoolMixerConfig)
class PoolMixer(nn.Module):
    def __init__(self, config: PoolMixerConfig):
        super().__init__()
        self.pool = nn.AvgPool2d(config.pool_size, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)
