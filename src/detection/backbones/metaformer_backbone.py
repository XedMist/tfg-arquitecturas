from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
from omegaconf import OmegaConf

log = logging.getLogger(__name__)

# ── Importaciones del proyecto ────────────────────────────────────────────────
# Añadir src/ al path para poder importar los módulos del proyecto
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from models.factory import (  # noqa: E402
    _build_dat_backbone,
    _build_gcnn_backbone,
    _build_mamba_backbone,
)
from models.metaformer import Metaformer  # noqa: E402

# ── Tabla de arquitecturas ────────────────────────────────────────────────────
# out_channels: canales de salida de cada stage (C2, C3, C4, C5)
# Los canales de C2 = in_dim del stage 0 (features pre-downsample)
_ARCH_SETTINGS: dict[str, dict] = {
    "gated_cnn": dict(
        builder="_build_gcnn_backbone",
        out_channels=[96, 192, 384, 576],
        depths=[3, 3, 9, 3],
    ),
    "gated_cnn_mamba": dict(
        builder="_build_mamba_backbone",
        out_channels=[96, 192, 384, 576],
        depths=[3, 3, 9, 3],
    ),
    "gated_cnn_dat": dict(
        builder="_build_dat_backbone",
        out_channels=[96, 192, 320, 512],
        depths=[3, 3, 9, 3],
    ),
}

_BUILDER_MAP = {
    "_build_gcnn_backbone": _build_gcnn_backbone,
    "_build_mamba_backbone": _build_mamba_backbone,
    "_build_dat_backbone": _build_dat_backbone,
}


class MetaFormerBackbone(nn.Module):
    """Backbone jerárquico MetaFormer compatible con MMDetection.

    Envuelve la arquitectura ``Metaformer`` entrenada en clasificación y la
    adapta para producir una pirámide de features multi-escala en formato
    BCHW, eliminando la cabeza de clasificación.

    Args:
        arch: Variante de arquitectura. Uno de ``'gated_cnn'``,
            ``'gated_cnn_mamba'``, ``'gated_cnn_dat'``.
        out_indices: Índices de stages a exponer (0=C2 /4, 1=C3 /8,
            2=C4 /16, 3=C5 /32). Por defecto todos.
        frozen_stages: Número de stages a congelar contando desde el stem.
            ``-1`` = nada congelado, ``0`` = solo stem, ``1`` = stem +
            stage 0, etc.
        pretrained: Ruta a un checkpoint en formato MMDetection (generado
            por ``convert_checkpoint.py``). Si ``None``, pesos aleatorios.
        num_classes: Necesario para instanciar el modelo base (se ignora
            la cabeza). Por defecto 1000.
        drop_path_rate: Stochastic depth rate del backbone.
    """

    def __init__(
        self,
        arch: str = "gated_cnn",
        out_indices: Sequence[int] = (0, 1, 2, 3),
        frozen_stages: int = -1,
        pretrained: str | None = None,
        num_classes: int = 1000,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()

        if arch not in _ARCH_SETTINGS:
            raise ValueError(
                f"arch={arch!r} desconocida. Opciones: {list(_ARCH_SETTINGS)}"
            )

        self.arch = arch
        self.out_indices = list(out_indices)
        self.frozen_stages = frozen_stages
        self.arch_cfg = _ARCH_SETTINGS[arch]

        # ── Construir modelo base ─────────────────────────────────────────────
        builder_name = self.arch_cfg["builder"]
        builder_fn = _BUILDER_MAP[builder_name]

        # OmegaConf minimal config para el builder
        cfg = OmegaConf.create(
            {
                "arch": arch.replace("_", "-"),  # factory usa "gated_cnn-mamba"
                "num_classes": num_classes,
                "pretrained": False,
                "drop_path_rate": drop_path_rate,
            }
        )
        base_model: Metaformer = builder_fn(cfg)  # type: ignore[assignment]

        # ── Separar componentes (sin cabeza) ──────────────────────────────────
        self.stem = base_model.stem
        self.stages = base_model.stages
        # La norma final es para clasificación global; en detección se omite
        # (cada FPN level tiene su propia normalización)

        # ── Cargar pesos preentrenados ────────────────────────────────────────
        if pretrained is not None:
            self._load_pretrained(pretrained)

        # ── Congelar stages ───────────────────────────────────────────────────
        self._freeze_stages()

    # ── API pública ──────────────────────────────────────────────────────────

    @property
    def out_channels(self) -> list[int]:
        """Canales de salida para cada índice en out_indices."""
        all_ch = self.arch_cfg["out_channels"]
        return [all_ch[i] for i in self.out_indices]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Forward pass produciendo la pirámide de features.

        Args:
            x: Tensor imagen en formato BCHW.

        Returns:
            Tupla de tensors BCHW, uno por cada índice en ``out_indices``.
            Resoluciones: /4, /8, /16, /32 para índices 0-3.
        """
        # stem: BCHW → BHWC, resolución /4
        x = self.stem(x)  # (B, H/4, W/4, C)

        outs: list[torch.Tensor] = []
        for i, stage in enumerate(self.stages):
            x, pre_down = stage(x, return_before_downsample=True)
            # pre_down: BHWC con resolución del stage actual
            if i in self.out_indices:
                # Convertir a BCHW para MMDetection
                outs.append(pre_down.permute(0, 3, 1, 2).contiguous())

        return tuple(outs)

    def train(self, mode: bool = True) -> "MetaFormerBackbone":
        """Override para mantener stages congelados en modo eval."""
        super().train(mode)
        self._freeze_stages()
        return self

    # ── Métodos internos ─────────────────────────────────────────────────────

    def _freeze_stages(self) -> None:
        """Congela stem y los primeros ``frozen_stages`` stages."""
        if self.frozen_stages >= 0:
            # Congelar stem
            self.stem.eval()
            for param in self.stem.parameters():
                param.requires_grad = False

        for i in range(min(self.frozen_stages, len(self.stages))):
            stage = self.stages[i]
            stage.eval()
            for param in stage.parameters():
                param.requires_grad = False

    def _load_pretrained(self, path: str) -> None:
        """Carga pesos desde un checkpoint en formato MMDetection.

        El checkpoint debe haber sido generado por ``convert_checkpoint.py``
        y contener ``{'state_dict': {...}}``.
        """
        log.info(f"Cargando pesos preentrenados desde: {path}")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt

        # Filtrar solo stem y stages (descartar head, norm global)
        backbone_state = {
            k: v
            for k, v in state_dict.items()
            if k.startswith("stem.") or k.startswith("stages.")
        }

        missing, unexpected = self.load_state_dict(backbone_state, strict=False)

        if missing:
            log.warning(f"Parámetros no cargados ({len(missing)}): {missing[:5]} ...")
        if unexpected:
            log.warning(
                f"Parámetros inesperados ignorados ({len(unexpected)}): "
                f"{unexpected[:5]} ..."
            )
        log.info(
            f"Checkpoint cargado: {len(backbone_state)} tensors, "
            f"{len(missing)} missing, {len(unexpected)} unexpected."
        )


# ── Registro MMDetection (opcional, solo si mmdet está instalado) ────────────
try:
    from mmdet.registry import MODELS  # type: ignore[import]

    MODELS.register_module(name="MetaFormerBackbone", module=MetaFormerBackbone)
    log.debug("MetaFormerBackbone registrado en mmdet MODELS registry.")
except ImportError:
    pass  # MMDetection no disponible; el wrapper sigue siendo usable standalone
