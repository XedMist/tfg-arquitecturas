from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence

import torch
import torch.nn as nn
from omegaconf import DictConfig
from timm.layers import (
    DropPath,
    LayerScale,
    Mlp,
    calculate_drop_path_rates,
    trunc_normal_,
)
from timm.layers.classifier import ClassifierHead

from models.mixer import BaseMixerConfig, build_mixer
from models.module import Downsample, Mlp
from models.module.stem import StemLayer

MixerClass = type[nn.Module]


@dataclass
class BlockConfig:
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    drop_path: float = 0.0
    layer_scale_init_value: Optional[float] = 1e-5
    use_mlp: bool = True
    skip_residual: bool = False


class MetaformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        mixer: nn.Module,
        cfg: Optional[BlockConfig] = None,
    ):
        super().__init__()
        if cfg is None:
            cfg = BlockConfig()

        self.use_mlp = cfg.use_mlp
        self.skip_residual = cfg.skip_residual
        ls_init = cfg.layer_scale_init_value

        self.norm1 = nn.LayerNorm(d_model)
        self.mixer = mixer
        self.ls1 = (
            LayerScale(d_model, init_values=ls_init) if ls_init else nn.Identity()
        )
        self.drop_path1 = (
            DropPath(cfg.drop_path) if cfg.drop_path > 0.0 else nn.Identity()
        )

        if self.use_mlp:
            self.norm2 = nn.LayerNorm(d_model)
            self.mlp = Mlp(d_model, mlp_ratio=cfg.mlp_ratio, dropout=cfg.dropout)
            self.ls2 = (
                LayerScale(d_model, init_values=ls_init) if ls_init else nn.Identity()
            )
            self.drop_path2 = (
                DropPath(cfg.drop_path) if cfg.drop_path > 0.0 else nn.Identity()
            )

    def _mixer_branch(self, x: torch.Tensor) -> torch.Tensor:
        res = self.drop_path1(self.ls1(self.mixer(self.norm1(x))))
        return res if self.skip_residual else x + res

    def _mlp_branch(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._mixer_branch(x)
        if self.use_mlp:
            x = self._mlp_branch(x)
        return x


@dataclass
class BlockSpec:
    mixer_class: MixerClass
    mixer_kwargs: dict = field(default_factory=dict)
    use_mlp: bool = True
    skip_residual: bool = False

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BlockSpec":
        mixer_kwargs = dict(d.get("mixer_kwargs", {}))
        return cls(
            mixer_class=d["mixer_class"],
            mixer_kwargs=mixer_kwargs,
            use_mlp=bool(mixer_kwargs.pop("use_mlp", True)),
            skip_residual=bool(mixer_kwargs.pop("skip_residual", False)),
        )


@dataclass
class StageConfig:
    in_dim: int
    out_dim: int
    mixer_configs: Sequence[BaseMixerConfig]
    block_cfg: BlockConfig = field(default_factory=BlockConfig)
    block_cfgs: Optional[Sequence[BlockConfig]] = None


class Stage(nn.Module):
    def __init__(self, cfg: StageConfig):
        super().__init__()
        self.in_dim = cfg.in_dim
        self.out_dim = cfg.out_dim

        blocks = []
        for i, mixer_cfg in enumerate(cfg.mixer_configs):
            mixer_module = build_mixer(mixer_cfg)

            if cfg.block_cfgs is not None and i < len(cfg.block_cfgs):
                block_config = cfg.block_cfgs[i]
            else:
                block_config = cfg.block_cfg

            block = MetaformerBlock(
                d_model=cfg.in_dim,
                mixer=mixer_module,
                cfg=block_config,
            )
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)
        self.downsample = (
            Downsample(cfg.in_dim, cfg.out_dim)
            if cfg.in_dim != cfg.out_dim
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        x = self.downsample(x)
        return x


@dataclass
class MetaFormerConfig:
    num_classes: int
    stages: List[StageConfig]


class Metaformer(nn.Module):
    def __init__(
        self,
        config: MetaFormerConfig,
    ):
        super().__init__()
        self.stem = StemLayer(out_channels=config.stages[0].in_dim)

        self.stages = nn.ModuleList([Stage(stage_cfg) for stage_cfg in config.stages])

        final_dim = config.stages[-1].out_dim

        self.norm = nn.LayerNorm(final_dim)

        self.head = ClassifierHead(
            final_dim,
            config.num_classes,
            pool_type="",
        )

        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = x.mean(dim=(1, 2))
        x = self.norm(x)
        return self.head(x)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
