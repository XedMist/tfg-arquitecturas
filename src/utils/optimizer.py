from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig

log = logging.getLogger(__name__)


def build_optimizer(cfg: DictConfig, model: nn.Module) -> torch.optim.Optimizer:
    opt_cfg = cfg.optimizer
    decay_params, no_decay_params = _split_param_groups(
        model,
        weight_decay=opt_cfg.weight_decay,
        layer_decay=opt_cfg.get("layer_decay"),
        base_lr=opt_cfg.lr,
    )

    name = opt_cfg.name.lower()
    log.info(
        f"Optimizador: {name} | lr={opt_cfg.lr} | "
        f"wd={opt_cfg.weight_decay} | "
        f"params con wd={sum(p.numel() for g in decay_params for p in g['params']):,} | "
        f"params sin wd={sum(p.numel() for g in no_decay_params for p in g['params']):,}"
    )

    param_groups = decay_params + no_decay_params

    if name == "adamw":
        return torch.optim.AdamW(
            param_groups,
            lr=opt_cfg.lr,
            betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
            eps=opt_cfg.get("eps", 1e-8),
        )
    elif name == "adam":
        return torch.optim.Adam(param_groups, lr=opt_cfg.lr)
    elif name == "sgd":
        return torch.optim.SGD(
            param_groups,
            lr=opt_cfg.lr,
            momentum=opt_cfg.get("momentum", 0.9),
            nesterov=opt_cfg.get("nesterov", True),
        )
    elif name == "lars":
        try:
            from torch.optim import LARS  # type: ignore
        except ImportError:
            raise ImportError("LARS requiere PyTorch >= 2.2 o paquete adicional.")
        return LARS(param_groups, lr=opt_cfg.lr)
    else:
        raise ValueError(f"Optimizador desconocido: {name}")


def _split_param_groups(
    model: nn.Module,
    weight_decay: float,
    layer_decay: Optional[float] = None,
    base_lr: float = 1e-3,
) -> tuple[list[dict], list[dict]]:
    decay, no_decay = [], []
    no_decay_names = {"bias", "norm"}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in no_decay_names) or param.ndim <= 1:
            no_decay.append(param)
        else:
            decay.append(param)

    decay_group = [{"params": decay, "weight_decay": weight_decay}]
    no_decay_group = [{"params": no_decay, "weight_decay": 0.0}]

    return decay_group, no_decay_group


def build_scheduler(
    cfg: DictConfig,
    optimizer: torch.optim.Optimizer,
    steps_per_epoch: Optional[int] = None,
):
    sched_cfg = cfg.scheduler
    name = sched_cfg.name.lower()
    total_epochs = cfg.training.epochs
    warmup_epochs = sched_cfg.get("warmup_epochs", 0)

    if name == "cosine":
        scheduler = _build_cosine_with_warmup(
            optimizer,
            total_epochs=total_epochs,
            warmup_epochs=warmup_epochs,
            min_lr=sched_cfg.get("min_lr", 1e-6),
            warmup_lr_init=sched_cfg.get("warmup_lr_init", 1e-6),
            base_lr=cfg.optimizer.lr,
        )
    elif name == "poly":
        scheduler = _build_poly_with_warmup(
            optimizer,
            total_epochs=total_epochs,
            warmup_epochs=warmup_epochs,
            power=sched_cfg.get("power", 0.9),
            min_lr=sched_cfg.get("min_lr", 1e-6),
        )
    elif name == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(sched_cfg.milestones),
            gamma=sched_cfg.get("gamma", 0.1),
        )
    elif name == "onecycle":
        if steps_per_epoch is None:
            raise ValueError("OneCycleLR requiere steps_per_epoch")
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.optimizer.lr,
            epochs=total_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=warmup_epochs / total_epochs,
        )
    else:
        raise ValueError(f"Scheduler desconocido: {name}")

    log.info(
        f"Scheduler: {name} | warmup_epochs={warmup_epochs} | total={total_epochs}"
    )
    return scheduler


def _build_cosine_with_warmup(
    optimizer, total_epochs, warmup_epochs, min_lr, warmup_lr_init, base_lr
):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Warmup lineal
            return max(warmup_lr_init / base_lr, epoch / max(warmup_epochs, 1))
        # Cosine decay
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return max(min_lr / base_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _build_poly_with_warmup(optimizer, total_epochs, warmup_epochs, power, min_lr):
    base_lr = optimizer.param_groups[0]["lr"]

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        factor = (1 - progress) ** power
        return max(min_lr / base_lr, factor)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def scale_lr(base_lr: float, batch_size: int, base_batch_size: int = 256) -> float:
    """
    Escala lineal del learning rate según batch size.
    Regla estándar: lr = base_lr * (batch_size / base_batch_size)
    (He et al., 2019 - "Bag of Tricks for Image Classification")
    """
    scaled = base_lr * batch_size / base_batch_size
    log.info(
        f"LR scaling: base_lr={base_lr} × ({batch_size}/{base_batch_size}) "
        f"= {scaled:.2e}"
    )
    return scaled
