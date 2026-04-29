from __future__ import annotations

import logging
import os
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from dataset.builder import build_classification_loaders
from models.factory import build_backbone
from utils.logger import ExperimentLogger, setup_logging
from utils.optimizer import build_optimizer, build_scheduler, scale_lr

log = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="classification", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # Logging
    output_dir = Path(f"outputs/{cfg.experiment.name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_file=output_dir / "train.log")
    log.info(f"\n{OmegaConf.to_yaml(cfg)}")

    # Seed

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        log.info(
            f"GPU: {torch.cuda.get_device_name(0)} | "
            + f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB",
        )
    else:
        log.warning("CUDA no disponible.")

    train_loader, val_loader = build_dataloaders(cfg)
    log.info(
        f"Dataset: {len(train_loader.dataset):,} train | {len(val_loader.dataset):,} val"
    )

    model = build_backbone(cfg).to(device)
    effective_bs = cfg.training.batch_size * cfg.training.grad_accumulation_steps
    cfg.optimizer.lr = scale_lr(cfg.optimizer.lr, effective_bs)

    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer, steps_per_epoch=len(train_loader))

    exp_logger = ExperimentLogger(cfg, output_dir)

    trainer = build_trainer(
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        exp_logger=exp_logger,
        device=device,
        ema=None,
    )

    trainer.fit()


def build_dataloaders(cfg: DictConfig):
    task = cfg.experiment.task
    return build_classification_loaders(cfg)


def build_trainer(
    cfg,
    model,
    optimizer,
    scheduler,
    train_loader,
    val_loader,
    exp_logger,
    device,
    ema,
):
    task = cfg.experiment.task
    common_args = dict(
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        exp_logger=exp_logger,
        device=device,
        ema=ema,
    )

    if task == "classification":
        from trainers.classification_trainer import ClassificationTrainer

        return ClassificationTrainer(**common_args)
    else:
        raise ValueError(f"Tarea desconocida: {task}")


if __name__ == "__main__":
    main()
