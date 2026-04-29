from dataclasses import dataclass

import torch
import torch.nn as nn

from models.mixer.mixer_config import BaseMixerConfig
from models.mixer.registry import register_mixer


@dataclass
class GatedCNNMixerConfig(BaseMixerConfig):
    expansion_ratio: float = 8 / 3
    kernel_size: int = 7
    conv_ratio: float = 1.0


@register_mixer(GatedCNNMixerConfig)
class GatedCNNMixer(nn.Module):
    def __init__(
        self,
        config: GatedCNNMixerConfig,
    ):
        super().__init__()
        hidden = int(config.expansion_ratio * config.d_model)
        self.fc1 = nn.Linear(config.d_model, hidden * 2)
        self.act = nn.GELU()
        conv_channels = int(config.conv_ratio * config.d_model)
        self.split_indices: list[int] = [hidden, hidden - conv_channels, conv_channels]
        self.conv = nn.Conv2d(
            conv_channels,
            conv_channels,
            kernel_size=config.kernel_size,
            padding=config.kernel_size // 2,
            groups=conv_channels,
        )
        self.fc2 = nn.Linear(hidden, config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=-1)
        c = c.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        c = self.conv(c)
        c = c.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        x = self.fc2(self.act(g) * torch.cat((i, c), dim=-1))
        return x
