from dataclasses import dataclass
import torch
import torch.nn as nn

from models.mixer.mixer_config import BaseMixerConfig
from models.mixer.registry import register_mixer


@dataclass
class IdentityMixerConfig(BaseMixerConfig):
    pass


@register_mixer(IdentityMixerConfig)
class IdentityMixer(nn.Module):
    def __init__(self, _cfg: IdentityMixerConfig):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
