from __future__ import annotations

import contextlib
import logging
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.amp.grad_scaler import GradScaler

from utils.logger import ExperimentLogger

log = logging.getLogger(__name__)


class BaseTrainer:
    """Abstract trainer.  Subclasses must implement :meth:`train_epoch`,
    :meth:`val_epoch` and :meth:`compute_loss`.
    """

    def __init__(
        self,
        cfg: DictConfig,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,  # any LRScheduler or timm scheduler
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        exp_logger: ExperimentLogger,
        device: str,
        ema=None,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = exp_logger
        self.device = device
        self.ema = ema

        # ── AMP ────────────────────────────────────────────────────────────
        # Prefer bfloat16 on Ampere+ (no loss scaling needed, numerically
        # stabler).  Fall back to float16 otherwise.  DeiT-III and Swin V2
        # both use bf16 when available.
        self._amp_enabled: bool = cfg.precision.amp
        self._amp_dtype: torch._C.dtype = self._resolve_amp_dtype(
            cfg.precision.get("dtype", "auto")
        )
        # GradScaler is a no-op when bfloat16 is used (no underflow risk),
        # but we keep it for fp16 and for the unified code path.
        self.scaler = GradScaler(
            device=self.device,
            enabled=self._amp_enabled and self._amp_dtype == torch._C.float16,
        )

        # ── Training hyper-params ─────────────────────────────────────────
        self.grad_accum: int = cfg.training.grad_accumulation_steps
        self.grad_clip: float = cfg.training.get("grad_clip_norm", 0.0)

        # ── State ─────────────────────────────────────────────────────────
        self.current_epoch: int = 0
        self.global_step: int = 0
        self.best_metric: float = float("-inf")
        self.best_checkpoint_path: Optional[Path] = None

        # ── Checkpointing ─────────────────────────────────────────────────
        self.ckpt_dir = Path(cfg.checkpoint.dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        if cfg.checkpoint.get("resume_from"):
            self._resume(cfg.checkpoint.resume_from)

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self) -> None:
        """Run the full training loop."""
        total_epochs: int = self.cfg.training.epochs
        log.info(
            "Starting training: %d epochs on %s | amp=%s dtype=%s",
            total_epochs,
            self.device,
            self._amp_enabled,
            self._amp_dtype,
        )
        start_time = time.monotonic()

        for epoch in range(self.current_epoch, total_epochs):
            self.current_epoch = epoch

            # ── Train ──────────────────────────────────────────────────────
            train_metrics = self.train_epoch(epoch)

            # ── Validate (with EMA weights when available) ─────────────────
            with self._ema_context():
                val_metrics = self.val_epoch(epoch)

            # ── Epoch-level LR step ────────────────────────────────────────
            # Only step here if the scheduler is epoch-based (e.g. MultiStep,
            # CosineAnnealingLR).  Step-based schedulers (timm CosineLRScheduler
            # with warmup) are stepped inside train_epoch via step_update().
            if self.scheduler is not None and not hasattr(
                self.scheduler, "step_update"
            ):
                self.scheduler.step(epoch + 1)

            # ── Logging ────────────────────────────────────────────────────
            all_metrics = {**train_metrics, **val_metrics, "epoch": epoch}
            self.logger.log(all_metrics, step=self.global_step)
            self.logger.log_metrics_csv(all_metrics)

            log_every = self.cfg.logging.console.get("log_every_n_epochs", 1)
            if (epoch + 1) % log_every == 0:
                self.logger.print_metrics_table(
                    {k: v for k, v in all_metrics.items() if isinstance(v, float)},
                    title=f"Epoch {epoch + 1}/{total_epochs}",
                )

            # ── Checkpoint ─────────────────────────────────────────────────
            monitor: str = self.cfg.checkpoint.monitor
            current_metric = val_metrics.get(
                monitor, val_metrics.get(monitor.split("/")[-1])
            )
            if current_metric is not None:
                self._save_checkpoint(epoch, current_metric)

        elapsed = time.monotonic() - start_time
        log.info("Training finished in %.2f h", elapsed / 3600)
        self.logger.finish()

    # ── Abstract methods ──────────────────────────────────────────────────────

    def train_epoch(self, epoch: int) -> dict[str, float]:
        raise NotImplementedError

    def val_epoch(self, epoch: int) -> dict[str, float]:
        raise NotImplementedError

    def compute_loss(self, batch) -> tuple[torch.Tensor, dict]:
        raise NotImplementedError

    # ── Protected helpers ─────────────────────────────────────────────────────

    def _forward_backward(
        self,
        batch,
        *,
        do_update: bool,
    ) -> tuple[float, dict]:
        """Single accumulation step.

        Args:
            batch:      The raw batch from the DataLoader.
            do_update:  Whether to apply the optimiser update this step
                        (i.e. ``global_step % grad_accum == 0``).

        Returns:
            Tuple of (unscaled loss value, metrics dict).
        """
        with torch.autocast(
            device_type=self.device,
            dtype=self._amp_dtype,
            enabled=self._amp_enabled,
        ):
            loss, metrics = self.compute_loss(batch)
            loss_scaled = loss / self.grad_accum

        self.scaler.scale(loss_scaled).backward()

        if do_update:
            # Must unscale *before* clipping so that the raw gradient
            # magnitudes are used — not the AMP-scaled ones.
            self.scaler.unscale_(self.optimizer)
            if self.grad_clip > 0.0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            if self.ema is not None:
                self.ema.update()

        return loss.item(), metrics

    @contextlib.contextmanager
    def _ema_context(self):
        """Context manager that activates EMA weights during validation."""
        if self.ema is not None:
            with self.ema.average_parameters():
                yield
        else:
            yield

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, metric: float) -> None:
        is_best = metric > self.best_metric
        if is_best:
            self.best_metric = metric

        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler else None
            ),
            "scaler_state_dict": self.scaler.state_dict(),
            "ema_state_dict": self.ema.state_dict() if self.ema else None,
            "best_metric": self.best_metric,
            "metric_value": metric,
            "cfg": self.cfg,
        }

        # Always keep the latest checkpoint for crash recovery.
        torch.save(state, self.ckpt_dir / "last.ckpt")

        if is_best:
            best_path = self.ckpt_dir / "best.ckpt"
            torch.save(state, best_path)
            self.best_checkpoint_path = best_path
            log.info("New best checkpoint  %.4f  →  %s", metric, best_path)

        epoch_path = self.ckpt_dir / f"epoch_{epoch:04d}_metric_{metric:.4f}.ckpt"
        torch.save(state, epoch_path)
        self._cleanup_checkpoints(keep_k=self.cfg.checkpoint.save_top_k)

    def _cleanup_checkpoints(self, keep_k: int = 3) -> None:
        """Delete epoch checkpoints beyond the top-k by metric value."""
        ckpts = sorted(
            self.ckpt_dir.glob("epoch_*.ckpt"),
            key=lambda p: float(p.stem.split("metric_")[-1]),
            reverse=True,
        )
        for old_ckpt in ckpts[keep_k:]:
            old_ckpt.unlink()
            log.debug("Removed checkpoint: %s", old_ckpt)

    def _resume(self, checkpoint_path: str) -> None:
        path = Path(checkpoint_path)
        if not path.exists():
            log.warning("Checkpoint not found at %s — starting from scratch.", path)
            return

        log.info("Resuming from checkpoint: %s", path)
        # weights_only=True is the safe default (PyTorch ≥ 2.0).
        state = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])

        if state.get("scheduler_state_dict") and self.scheduler is not None:
            self.scheduler.load_state_dict(state["scheduler_state_dict"])

        if state.get("scaler_state_dict"):
            self.scaler.load_state_dict(state["scaler_state_dict"])

        if state.get("ema_state_dict") and self.ema is not None:
            self.ema.load_state_dict(state["ema_state_dict"])

        self.current_epoch = state["epoch"] + 1
        self.global_step = state.get("global_step", 0)
        self.best_metric = state.get("best_metric", float("-inf"))
        log.info(
            "Resumed at epoch %d  (best metric so far: %.4f)",
            self.current_epoch,
            self.best_metric,
        )

    # ── Private utilities ─────────────────────────────────────────────────────

    @staticmethod
    def _resolve_amp_dtype(dtype_str: str) -> torch._C.dtype:
        """Return the AMP compute dtype.

        "auto"    → bf16 on Ampere+ (sm_80+), fp16 otherwise.
        "bfloat16"→ torch.bfloat16 unconditionally.
        "float16" → torch.float16 unconditionally.
        """
        if dtype_str == "bfloat16":
            return torch._C.bfloat16
        if dtype_str == "float16":
            return torch._C.float16
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            return torch._C.bfloat16
        return torch._C.float16
