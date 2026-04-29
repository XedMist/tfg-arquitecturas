from __future__ import annotations

import logging
from typing import Optional

import timm
import torch
import torch.nn as nn
from omegaconf import DictConfig
from timm.layers import calculate_drop_path_rates

from models.metaformer import BlockConfig, Metaformer, MetaFormerConfig, StageConfig
from models.mixer.gated_cnn import GatedCNNMixerConfig

log = logging.getLogger(__name__)


def build_backbone(cfg: DictConfig) -> nn.Module:
    model_cfg = cfg.model
    task = cfg.experiment.task

    log.info(
        f"Construyendo backbone: arch={model_cfg.arch}, "
        f"pretrained={model_cfg.pretrained}, task={task}"
    )

    if task == "classification":
        model = _build_classifier(model_cfg)
    else:
        raise ValueError(f"Tarea desconocida: {task}")

    if cfg.precision.get("compile", False):
        mode = cfg.precision.get("compile_mode", "reduce-overhead")
        log.info(f"Compilando modelo con torch.compile (mode={mode})...")
        model = torch.compile(model, mode=mode)
        log.info("torch.compile completado.")

    _log_model_info(model)
    return model


def _build_classifier(cfg: DictConfig) -> nn.Module:
    drop_path_rate = cfg.get("drop_path_rate", 0.0)
    depths = [3, 3, 9, 3]
    dp_rates = calculate_drop_path_rates(drop_path_rate, sum(depths))

    config = MetaFormerConfig(
        num_classes=10,
        stages=[
            StageConfig(
                in_dim=96,
                out_dim=192,
                mixer_configs=[GatedCNNMixerConfig(48)] * depths[0],
                block_cfgs=[
                    BlockConfig(use_mlp=False, drop_path=dp_rates[i])
                    for i in range(sum(depths[:0]), sum(depths[:1]))
                ],
            ),
            StageConfig(
                in_dim=192,
                out_dim=384,
                mixer_configs=[GatedCNNMixerConfig(96)] * depths[1],
                block_cfgs=[
                    BlockConfig(use_mlp=False, drop_path=dp_rates[i])
                    for i in range(sum(depths[:1]), sum(depths[:2]))
                ],
            ),
            StageConfig(
                in_dim=384,
                out_dim=576,
                mixer_configs=[GatedCNNMixerConfig(192)] * depths[2],
                block_cfgs=[
                    BlockConfig(use_mlp=False, drop_path=dp_rates[i])
                    for i in range(sum(depths[:2]), sum(depths[:3]))
                ],
            ),
            StageConfig(
                in_dim=576,
                out_dim=576,
                mixer_configs=[GatedCNNMixerConfig(288)] * depths[3],
                block_cfgs=[
                    BlockConfig(use_mlp=False, drop_path=dp_rates[i])
                    for i in range(sum(depths[:3]), sum(depths[:4]))
                ],
            ),
        ],
    )
    model = Metaformer(config)

    return model


def _log_model_info(model: nn.Module) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    size_mb = total * 4 / 1024 / 1024  # fp32

    log.info(
        f"Modelo: {total:,} params totales | "
        f"{trainable:,} entrenables | "
        f"{frozen:,} congelados | "
        f"~{size_mb:.1f} MB (fp32)"
    )
