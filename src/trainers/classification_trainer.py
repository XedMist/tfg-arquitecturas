from __future__ import annotations

import logging
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torchmetrics.classification import MulticlassAccuracy

from trainers.base import BaseTrainer
from utils.logger import ExperimentLogger

log = logging.getLogger(__name__)


class ClassificationTrainer(BaseTrainer):
    def __init__(
        self,
        cfg: DictConfig,
        model,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        exp_logger: ExperimentLogger,
        device,
        ema=None,
    ):
        super().__init__(
            cfg,
            model,
            optimizer,
            scheduler,
            train_loader,
            val_loader,
            exp_logger,
            device,
            ema,
        )
        nc = cfg.model.num_classes
        self.train_acc1 = MulticlassAccuracy(num_classes=nc, top_k=1).to(device)
        self.train_acc5 = MulticlassAccuracy(num_classes=nc, top_k=5).to(device)
        self.val_acc1 = MulticlassAccuracy(num_classes=nc, top_k=1).to(device)
        self.val_acc5 = MulticlassAccuracy(num_classes=nc, top_k=5).to(device)

        aug_cfg = cfg.augmentation.train
        self.mixup_alpha = aug_cfg.get("mixup_alpha", 0.0)
        self.cutmix_alpha = aug_cfg.get("cutmix_alpha", 0.0)
        self.label_smooth = 0.1

    def train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        self.train_acc1.reset()
        self.train_acc5.reset()

        total_loss = 0.0
        n_batches = len(self.train_loader)

        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            images, targets_a, targets_b, lam = self._apply_mixing(images, targets)

            accumulate = (batch_idx + 1) % self.grad_accum != 0

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=self.cfg.precision.amp,
            ):
                logits = self.model(images)
                if targets_a is not None:
                    loss = lam * self._cross_entropy(logits, targets_a) + (
                        1 - lam
                    ) * self._cross_entropy(logits, targets_b)
                else:
                    loss = self._cross_entropy(logits, targets)
                loss_scaled = loss / self.grad_accum

            self.scaler.scale(loss_scaled).backward()

            if not accumulate:
                self.scaler.unscale_(self.optimizer)
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                if self.ema:
                    self.ema.update()
                self.global_step += 1

            total_loss += loss.item()

            # Métricas de train (sin MixUp para interpretabilidad)
            with torch.no_grad():
                self.train_acc1.update(logits, targets)
                self.train_acc5.update(logits, targets)

            # Log por step
            if self.global_step % self.cfg.logging.console.log_every_n_steps == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                self.logger.log(
                    {
                        "train/loss_step": loss.item(),
                        "train/lr": lr,
                    },
                    step=self.global_step,
                    commit=False,
                )

        acc1 = self.train_acc1.compute().item()
        acc5 = self.train_acc5.compute().item()
        avg_loss = total_loss / n_batches

        # Actualizar LR scheduler por step si usa warmup
        if hasattr(self.scheduler, "step_update"):
            self.scheduler.step_update(epoch * n_batches + n_batches)

        return {
            "train/loss": avg_loss,
            "train/acc_top1": acc1,
            "train/acc_top5": acc5,
            "train/lr": self.optimizer.param_groups[0]["lr"],
        }

    # ── Val epoch ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def val_epoch(self, epoch: int) -> dict[str, float]:
        self.model.eval()
        self.val_acc1.reset()
        self.val_acc5.reset()
        total_loss = 0.0

        for images, targets in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=self.cfg.precision.amp,
            ):
                logits = self.model(images)
                loss = F.cross_entropy(logits, targets)

            total_loss += loss.item()
            self.val_acc1.update(logits, targets)
            self.val_acc5.update(logits, targets)

        return {
            "val/loss": total_loss / len(self.val_loader),
            "val/acc_top1": self.val_acc1.compute().item(),
            "val/acc_top5": self.val_acc5.compute().item(),
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def compute_loss(self, batch):
        """Implementación requerida por BaseTrainer."""
        images, targets = batch
        logits = self.model(images)
        loss = self._cross_entropy(logits, targets)
        return loss, {}

    def _cross_entropy(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        return F.cross_entropy(logits, targets, label_smoothing=self.label_smooth)

    def _apply_mixing(self, images, targets):
        """Aplica MixUp o CutMix aleatoriamente si alpha > 0."""
        r = random.random()
        targets_a = targets_b = None
        lam = 1.0

        use_mixup = self.mixup_alpha > 0 and r < 0.5
        use_cutmix = self.cutmix_alpha > 0 and r >= 0.5

        if use_mixup:
            import numpy as np

            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            batch_size = images.size(0)
            rand_idx = torch.randperm(batch_size, device=images.device)
            targets_a, targets_b = targets, targets[rand_idx]
            images = lam * images + (1 - lam) * images[rand_idx]

        elif use_cutmix:
            import numpy as np

            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            batch_size = images.size(0)
            rand_idx = torch.randperm(batch_size, device=images.device)
            targets_a, targets_b = targets, targets[rand_idx]

            bbx1, bby1, bbx2, bby2 = self._rand_bbox(images.size(), lam)
            images[:, :, bbx1:bbx2, bby1:bby2] = images[
                rand_idx, :, bbx1:bbx2, bby1:bby2
            ]
            lam = 1 - (
                (bbx2 - bbx1) * (bby2 - bby1) / (images.size(-1) * images.size(-2))
            )

        return images, targets_a, targets_b, lam

    @staticmethod
    def _rand_bbox(size, lam):
        import numpy as np

        W, H = size[2], size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        return bbx1, bby1, bbx2, bby2
