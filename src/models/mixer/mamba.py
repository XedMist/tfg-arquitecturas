from dataclasses import dataclass

import torch
import torch.nn as nn

from models.mixer.mixer_config import BaseMixerConfig
from models.mixer.registry import register_mixer

try:
    from mamba_ssm.modules.mamba_simple import Mamba
except ImportError:
    Mamba: type | None = None


@dataclass
class MambaMixerConfig(BaseMixerConfig):
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    bimamba_type: str = "v2"


@register_mixer(MambaMixerConfig)
class MambaMixer(nn.Module):
    def __init__(
        self,
        config: MambaMixerConfig,
    ):
        super().__init__()
        if Mamba is None:
            raise ImportError(
                "mamba_ssm is not installed. Install with: pip install mamba-ssm"
            )
        self.inner = Mamba(  # type: ignore[operator]
            d_model=config.d_model,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
            bimamba_type=config.bimamba_type,
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B, H, W, C = X.shape
        X = X.view(B, H * W, C)
        X = self.inner(X)
        return X.view(B, H, W, C)
