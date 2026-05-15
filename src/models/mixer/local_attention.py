import torch
import torch.nn as nn
import einops
from models.mixer.mixer_config import BaseMixerConfig
from dataclasses import dataclass

from models.mixer.registry import register_mixer


@dataclass
class LocalAttentionMixerConfig(BaseMixerConfig):
    num_heads: int = 32
    attn_drop: float = 0.0
    proj_drop: float = 0.0


@register_mixer(LocalAttentionMixerConfig)
class LocalAttentionMixer(nn.Module):
    def __init__(self, config: LocalAttentionMixerConfig):
        super().__init__()
        self.num_heads = config.num_heads
        assert (
            config.d_model % config.num_heads == 0
        ), f"D({config.d_model}) is not divisible by H({config.num_heads})"
        head_dim = config.d_model // config.num_heads

        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(config.d_model, config.d_model * 3)
        self.attn_drop = nn.Dropout(config.attn_drop)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.proj_drop = nn.Dropout(config.proj_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B, C, H, W = X.shape
        X = einops.rearrange(X, "b c h w -> b (h w) c")

        qkv: torch.Tensor = (
            self.qkv(X)
            .reshape(B, H * W, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )

        Q, K, V = qkv[0], qkv[1], qkv[2]

        Q = Q * self.scale

        attn: torch.Tensor = Q @ K.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        Y = (attn @ V).transpose(1, 2).reshape(B, H * W, C)
        Y = self.out_proj(Y)
        Y = self.proj_drop(Y)

        return Y
