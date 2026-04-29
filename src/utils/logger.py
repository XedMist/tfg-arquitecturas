from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional, Union

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich import print as rprint

def setup_logging(level: str = "INFO", log_file: Optional[Path] = None) -> None:
    handlers: list[logging.Handler] = [
        RichHandler(
            console=Console(stderr=True),
            show_time=True,
            show_path=True,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
            markup=True,
        )
    ]

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        handlers.append(file_handler)

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
        force=True,
    )

    # Silenciar loggers verbosos de terceros
    for noisy in ("PIL", "matplotlib", "urllib3", "filelock", "timm"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


class ExperimentLogger:
    def __init__(self, cfg: Any, output_dir: Path) -> None:
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._log = logging.getLogger(self.__class__.__name__)

        self._wandb = None
        self._tb_writer = None
        self._step = 0


    def log(
        self,
        metrics: dict[str, Any],
        step: Optional[int] = None,
        commit: bool = True,
    ) -> None:
        step = step if step is not None else self._step
        if commit:
            self._step = step

        if self._wandb is not None:
            self._wandb.log(metrics, step=step, commit=commit)

        if self._tb_writer is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self._tb_writer.add_scalar(k, v, global_step=step)

    def log_image(
        self,
        key: str,
        images: Any,
        step: Optional[int] = None,
        caption: Optional[str] = None,
    ) -> None:
        step = step if step is not None else self._step
        if self._wandb is not None:
            self._wandb.log(
                {key: self._wandb.Image(images, caption=caption)},
                step=step,
            )
        if self._tb_writer is not None:
            import torch
            if isinstance(images, torch.Tensor) and images.dim() == 4:
                from torchvision.utils import make_grid
                grid = make_grid(images[:8], normalize=True)
                self._tb_writer.add_image(key, grid, global_step=step)

    def log_confusion_matrix(
        self, y_true: Any, y_pred: Any, class_names: list[str], step: int
    ) -> None:
        if self._wandb is not None:
            self._wandb.log(
                {
                    "confusion_matrix": self._wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=y_true,
                        preds=y_pred,
                        class_names=class_names,
                    )
                },
                step=step,
            )

    def log_model(self, model_path: Path, name: str = "backbone") -> None:
        if self._wandb is not None:
            artifact = self._wandb.Artifact(name, type="model")
            artifact.add_file(str(model_path))
            self._wandb.log_artifact(artifact)

    def print_metrics_table(self, metrics: dict[str, float], title: str = "Métricas") -> None:
        console = Console()
        table = Table(title=title, show_header=True, header_style="bold cyan")
        table.add_column("Métrica", style="dim", width=30)
        table.add_column("Valor", justify="right")

        for k, v in sorted(metrics.items()):
            color = "green" if "acc" in k or "iou" in k or "map" in k.lower() else "white"
            table.add_row(k, f"[{color}]{v:.4f}[/{color}]")

        console.print(table)

    def finish(self) -> None:
        if self._tb_writer is not None:
            self._tb_writer.close()
        if self._wandb is not None:
            self._wandb.finish()
        self._log.info("Logger cerrado correctamente.")



def make_progress_bar() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        expand=True,
    )
