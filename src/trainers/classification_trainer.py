from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torchmetrics.classification import MulticlassAccuracy

from trainers.base import BaseTrainer
from utils.logger import ExperimentLogger, make_progress_bar

log = logging.getLogger(__name__)


class ClassificationTrainer(BaseTrainer):
    def __init__(
        self,
        cfg: DictConfig,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        exp_logger: ExperimentLogger,
        device: str,
        ema=None,
    ) -> None:
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

        nc: int = cfg.model.num_classes

        # torchmetrics objects — one set per split to avoid state bleeding.
        self.train_acc1 = MulticlassAccuracy(num_classes=nc, top_k=1).to(device)
        self.train_acc5 = MulticlassAccuracy(num_classes=nc, top_k=5).to(device)
        self.val_acc1 = MulticlassAccuracy(num_classes=nc, top_k=1).to(device)
        self.val_acc5 = MulticlassAccuracy(num_classes=nc, top_k=5).to(device)

        aug_cfg = cfg.augmentation.train
        self.mixup_alpha: float = aug_cfg.get("mixup_alpha", 0.0)
        self.cutmix_alpha: float = aug_cfg.get("cutmix_alpha", 0.0)

        # Label smoothing ε — 0.1 is the DeiT default.
        self.label_smooth: float = cfg.training.get("label_smoothing", 0.1)

        # Whether any mixing augmentation is active.
        self._use_mixing: bool = self.mixup_alpha > 0 or self.cutmix_alpha > 0

    # ── Train epoch ───────────────────────────────────────────────────────────

    def train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        self.train_acc1.reset()
        self.train_acc5.reset()

        total_loss = 0.0
        n_batches = len(self.train_loader)

        progress = make_progress_bar()
        task_id = progress.add_task(f"Epoch {epoch + 1}", total=n_batches)

        with progress:
            for batch_idx, (images, targets) in enumerate(self.train_loader):
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                # ── Mixing augmentations ────────────────────────────────────
                mixed_images, targets_a, targets_b, lam = self._apply_mixing(
                    images, targets
                )

                # ── Forward + backward (handles grad accumulation) ──────────
                do_update = (batch_idx + 1) % self.grad_accum == 0 or (
                    batch_idx + 1 == n_batches
                )

                with torch.autocast(
                    device_type=self.device,
                    dtype=self._amp_dtype,
                    enabled=self._amp_enabled,
                ):
                    logits = self.model(mixed_images)
                    loss = self._mixing_loss(logits, targets, targets_a, targets_b, lam)
                    loss_scaled = loss / self.grad_accum

                self.scaler.scale(loss_scaled).backward()

                if do_update:
                    self.scaler.unscale_(self.optimizer)
                    if self.grad_clip > 0.0:
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.grad_clip
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.ema is not None:
                        self.ema.update()
                    self.global_step += 1

                    # Step-level LR scheduler (timm CosineLRScheduler).
                    if hasattr(self.scheduler, "step_update"):
                        self.scheduler.step_update(num_updates=self.global_step)

                total_loss += loss.item()

                # Accuracy is always measured on the original (unmixed) targets
                # so the metric is comparable across all epochs — including
                # those without mixing.
                with torch.no_grad():
                    self.train_acc1.update(logits.detach(), targets)
                    self.train_acc5.update(logits.detach(), targets)

                # ── Step-level logging ──────────────────────────────────────
                log_every: int = self.cfg.logging.console.log_every_n_steps
                if self.global_step % log_every == 0:
                    self.logger.log(
                        {
                            "train/loss_step": loss.item(),
                            "train/lr": self._current_lr(),
                        },
                        step=self.global_step,
                        commit=False,
                    )

                progress.update(task_id, advance=1)

        return {
            "train/loss": total_loss / n_batches,
            "train/acc_top1": self.train_acc1.compute().item(),
            "train/acc_top5": self.train_acc5.compute().item(),
            "train/lr": self._current_lr(),
        }

    # ── Val epoch ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def val_epoch(self, epoch: int) -> dict[str, float]:
        """Evaluate on the validation split.

        No mixing, no label smoothing — raw cross-entropy on hard targets.
        This matches the DeiT / timm evaluation protocol.
        """
        self.model.eval()
        self.val_acc1.reset()
        self.val_acc5.reset()

        total_loss = 0.0
        n_batches = len(self.val_loader)

        progress = make_progress_bar()
        task_id = progress.add_task(f"Val Epoch {epoch + 1}", total=n_batches)

        with progress:
            for images, targets in self.val_loader:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                with torch.autocast(
                    device_type=self.device,
                    dtype=self._amp_dtype,
                    enabled=self._amp_enabled,
                ):
                    logits = self.model(images)
                    # Validation uses plain cross-entropy (no label smoothing)
                    # so the loss value is directly interpretable.
                    loss = F.cross_entropy(logits, targets)

                total_loss += loss.item()
                self.val_acc1.update(logits, targets)
                self.val_acc5.update(logits, targets)

                progress.update(task_id, advance=1)

        return {
            "val/loss": total_loss / n_batches,
            "val/acc_top1": self.val_acc1.compute().item(),
            "val/acc_top5": self.val_acc5.compute().item(),
        }

    # ── compute_loss (BaseTrainer contract) ───────────────────────────────────

    def compute_loss(self, batch) -> tuple[torch.Tensor, dict]:
        """Simple forward for use by BaseTrainer._forward_backward if ever
        called without mixing (e.g. by a sub-class).
        """
        images, targets = batch
        logits = self.model(images)
        loss = self._ce(logits, targets)
        return loss, {}

    # ── Private helpers ───────────────────────────────────────────────────────

    def _mixing_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        targets_a: Optional[torch.Tensor],
        targets_b: Optional[torch.Tensor],
        lam: float,
    ) -> torch.Tensor:
        """Return the (possibly mixed) loss.

        When mixing is inactive (no mixing this batch), targets_a is None
        and we fall back to the standard cross-entropy.
        """
        if targets_a is not None:
            # Mixed loss: E[λ·CE(a) + (1-λ)·CE(b)]
            return lam * self._ce(logits, targets_a) + (1.0 - lam) * self._ce(
                logits, targets_b
            )
        return self._ce(logits, targets)

    def _ce(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, targets, label_smoothing=self.label_smooth)

    def _apply_mixing(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], float]:
        """Apply MixUp or CutMix (timm-style selection).

        Returns:
            (mixed_images, targets_a, targets_b, lam)
            targets_a / targets_b are None when no mixing is applied.
        """
        if not self._use_mixing:
            return images, None, None, 1.0

        # Choose MixUp vs CutMix with equal probability when both are enabled.
        use_cutmix = self.cutmix_alpha > 0 and (
            self.mixup_alpha <= 0 or torch.rand(1).item() > 0.5
        )
        alpha = self.cutmix_alpha if use_cutmix else self.mixup_alpha
        lam: float = float(np.random.beta(alpha, alpha))

        batch_size = images.size(0)
        rand_idx = torch.randperm(batch_size, device=images.device)
        targets_a = targets
        targets_b = targets[rand_idx]

        if use_cutmix:
            images, lam = self._cutmix_images(images, rand_idx, lam)
        else:
            images = lam * images + (1.0 - lam) * images[rand_idx]

        return images, targets_a, targets_b, lam

    @staticmethod
    def _cutmix_images(
        images: torch.Tensor,
        rand_idx: torch.Tensor,
        lam: float,
    ) -> tuple[torch.Tensor, float]:
        """Paste a random crop from the shuffled batch into the original.

        Returns the patched image tensor and the recomputed λ (actual area ratio).
        """
        _, _, H, W = images.shape
        cut_rat = np.sqrt(1.0 - lam)
        cut_h = int(H * cut_rat)
        cut_w = int(W * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        x1 = int(np.clip(cx - cut_w // 2, 0, W))
        y1 = int(np.clip(cy - cut_h // 2, 0, H))
        x2 = int(np.clip(cx + cut_w // 2, 0, W))
        y2 = int(np.clip(cy + cut_h // 2, 0, H))

        images = images.clone()
        images[:, :, y1:y2, x1:x2] = images[rand_idx, :, y1:y2, x1:x2]

        # Recompute λ from the actual pasted area so the loss weighting is exact.
        lam = 1.0 - (x2 - x1) * (y2 - y1) / (W * H)
        return images, lam

    def _current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]
