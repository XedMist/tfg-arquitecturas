from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.amp import GradScaler

from utils.logger import ExperimentLogger, make_progress_bar

log = logging.getLogger(__name__)


class BaseTrainer:
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
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = exp_logger
        self.device = device
        self.ema = ema

        self.scaler = GradScaler(device="cuda", enabled=cfg.precision.amp)
        self.grad_accum = cfg.training.grad_accumulation_steps
        self.grad_clip = cfg.training.get("grad_clip_norm", 0.0)

        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float("-inf")
        self.best_checkpoint_path: Optional[Path] = None

        self.ckpt_dir = Path(cfg.checkpoint.dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        if cfg.checkpoint.get("resume_from"):
            self._resume(cfg.checkpoint.resume_from)

    def fit(self) -> None:
        total_epochs = self.cfg.training.epochs
        log.info(f"Iniciando entrenamiento: {total_epochs} épocas en {self.device}")

        start_time = time.time()

        for epoch in range(self.current_epoch, total_epochs):
            self.current_epoch = epoch

            # ── Train ──
            train_metrics = self.train_epoch(epoch)

            # ── Validate ──
            with self.ema_context():
                val_metrics = self.val_epoch(epoch)

            # ── LR step ──
            if self.scheduler is not None:
                if hasattr(self.scheduler, "step"):
                    self.scheduler.step(epoch + 1)

            # ── Log ──
            all_metrics = {**train_metrics, **val_metrics, "epoch": epoch}
            self.logger.log(all_metrics, step=self.global_step)

            if (epoch + 1) % self.cfg.logging.console.get("log_every_n_steps", 1) == 0:
                self.logger.print_metrics_table(
                    {k: v for k, v in all_metrics.items() if isinstance(v, float)},
                    title=f"Época {epoch + 1}/{total_epochs}",
                )

            # ── Checkpoint ──
            monitor = self.cfg.checkpoint.monitor
            current_metric = val_metrics.get(
                monitor, val_metrics.get(monitor.split("/")[-1], None)
            )
            if current_metric is not None:
                self._save_checkpoint(epoch, current_metric)

        elapsed = time.time() - start_time
        log.info(f"Entrenamiento completado en {elapsed / 3600:.2f}h")
        self.logger.finish()

    def train_epoch(self, epoch: int) -> dict[str, float]:
        raise NotImplementedError

    def val_epoch(self, epoch: int) -> dict[str, float]:
        raise NotImplementedError

    def _forward_step(self, batch, accumulate: bool = False):
        with torch.autocast(
            device_type="cuda", dtype=torch._C.float16, enabled=self.cfg.precision.amp
        ):
            loss, metrics = self.compute_loss(batch)
            loss = loss / self.grad_accum

        self.scaler.scale(loss).backward()

        if not accumulate:
            # Unscale antes de grad clip
            self.scaler.unscale_(self.optimizer)
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)  # más eficiente que zero_grad()

            if self.ema is not None:
                self.ema.update()

        return loss.item() * self.grad_accum, metrics

    def compute_loss(self, batch) -> tuple[torch.Tensor, dict]:
        raise NotImplementedError

    def ema_context(self):
        if self.ema is not None:
            return self.ema.average_parameters()
        return _null_context()

    def _save_checkpoint(self, epoch: int, metric: float) -> None:
        is_best = metric > self.best_metric
        if is_best:
            self.best_metric = metric

        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict()
            if self.scheduler
            else None,
            "scaler_state_dict": self.scaler.state_dict(),
            "ema_state_dict": self.ema.state_dict() if self.ema else None,
            "best_metric": self.best_metric,
            "metric_value": metric,
            "cfg": self.cfg,
        }

        # Guardar last
        last_path = self.ckpt_dir / "last.ckpt"
        torch.save(state, last_path)

        # Guardar best
        if is_best:
            best_path = self.ckpt_dir / "best.ckpt"
            torch.save(state, best_path)
            self.best_checkpoint_path = best_path
            log.info(
                f"[bold green]Nuevo mejor checkpoint: {metric:.4f}[/bold green] → {best_path}",
                extra={"markup": True},
            )

        # Guardar epoch checkpoint
        epoch_path = self.ckpt_dir / f"epoch_{epoch:04d}_metric_{metric:.4f}.ckpt"
        torch.save(state, epoch_path)

        # Limpiar checkpoints viejos (top-k)
        self._cleanup_checkpoints(keep_k=self.cfg.checkpoint.save_top_k)

    def _cleanup_checkpoints(self, keep_k: int = 3) -> None:
        ckpts = sorted(
            [p for p in self.ckpt_dir.glob("epoch_*.ckpt")],
            key=lambda p: float(p.stem.split("metric_")[-1]),
            reverse=True,
        )
        for old_ckpt in ckpts[keep_k:]:
            old_ckpt.unlink()
            log.debug(f"Checkpoint eliminado: {old_ckpt}")

    def _resume(self, checkpoint_path: str) -> None:
        path = Path(checkpoint_path)
        if not path.exists():
            log.warning(f"Checkpoint no encontrado: {path}. Iniciando desde cero.")
            return

        log.info(f"Reanudando desde checkpoint: {path}")
        state = torch.load(path, map_location=self.device)

        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        if state.get("scheduler_state_dict") and self.scheduler:
            self.scheduler.load_state_dict(state["scheduler_state_dict"])
        if state.get("scaler_state_dict"):
            self.scaler.load_state_dict(state["scaler_state_dict"])
        if state.get("ema_state_dict") and self.ema:
            self.ema.load_state_dict(state["ema_state_dict"])

        self.current_epoch = state["epoch"] + 1
        self.global_step = state.get("global_step", 0)
        self.best_metric = state.get("best_metric", float("-inf"))
        log.info(f"Reanudando desde época {self.current_epoch}")


class _null_context:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
