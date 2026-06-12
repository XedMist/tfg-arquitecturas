"""
Backbone wrappers custom para Mask2Former (Detectron2).

- ``MetaformerBackbone``  → wrapper para los 3 metaformers custom
  (``gated_cnn``, ``gated_cnn-dat``, ``gated_cnn-mamba``).
- ``ResNet50Backbone``    → wrapper para ResNet50 de torchvision, que carga
  pesos desde un checkpoint de clasificación custom (no el ImageNet
  preentrenado de detectron2).

Comparten ``load_pretrained_into``: aborta con RuntimeError ante shape
mismatch o falta de pesos críticos (stem/stages para metaformer,
conv1/layer para resnet), y avisa ante pesos no críticos faltantes o
inesperados.
"""

from __future__ import annotations

from typing import Dict, List, Set

import torch
import torch.nn as nn

from detectron2.config import CfgNode as CN
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec

from models.metaformer import Metaformer


def add_metaformer_config(cfg: CN) -> None:
    """Añade los nodos ``MODEL.METAFORMER.*``.

    Tanto ``MetaformerBackbone`` como ``ResNet50Backbone`` leen el path de
    pesos de ``cfg.MODEL.METAFORMER.WEIGHTS`` para mantener un único
    punto de configuración.
    """
    cfg.MODEL.METAFORMER = CN()
    cfg.MODEL.METAFORMER.ARCH_NAME = "gated_cnn"
    cfg.MODEL.METAFORMER.WEIGHTS = ""
    cfg.MODEL.METAFORMER.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.METAFORMER.OUT_CHANNELS = [96, 192, 384, 768]
    cfg.MODEL.METAFORMER.OUT_STRIDES = [4, 8, 16, 32]
    cfg.SOLVER.BACKBONE_LR_MULTIPLIER = 0.1


def load_pretrained_into(
    model: nn.Module,
    path: str,
    expected_missing: Set[str],
    arch_name: str,
    critical_prefixes: tuple,
    state_dict: dict = None,
) -> None:
    """Carga pesos pre-entrenados desde un checkpoint de clasificación.

    - Shapes compatibles → copia in-place.
    - Shape mismatch → RuntimeError.
    - Clave del modelo ausente y fuera de ``expected_missing`` y con
      prefijo crítico → RuntimeError.
    - Otros casos (claves del ckpt no usadas, o pesos del modelo no
      críticos ausentes) → solo advertencia.

    Si se pasa ``state_dict`` explícitamente, se salta la carga desde
    ``path`` y se usa directamente. Útil cuando el caller quiere
    pre-procesar las keys (p. ej. añadir un prefijo).
    """
    if state_dict is None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = checkpoint
        for key in ("model", "state_dict", "model_state_dict"):
            if isinstance(checkpoint, dict) and key in checkpoint:
                state_dict = checkpoint[key]
                break

    own_state = model.state_dict()

    matched: List[str] = []
    mismatched: List[tuple] = []
    unexpected: List[str] = []
    missing_unexpected: List[str] = []

    for k, v in state_dict.items():
        if k not in own_state:
            unexpected.append(k)
            continue
        if tuple(own_state[k].shape) != tuple(v.shape):
            mismatched.append((k, tuple(v.shape), tuple(own_state[k].shape)))
            continue
        own_state[k].copy_(v)
        matched.append(k)

    for k in own_state:
        if k not in state_dict and k not in expected_missing:
            missing_unexpected.append(k)

    cls_name = type(model).__name__

    if mismatched:
        lines = [
            f"[{cls_name}] ABORT: shape mismatch en {len(mismatched)} tensores "
            f"(arch={arch_name}, ckpt={path}):"
        ]
        for k, ckpt_shape, model_shape in mismatched[:10]:
            lines.append(f"  {k}: checkpoint={ckpt_shape} vs model={model_shape}")
        if len(mismatched) > 10:
            lines.append(f"  ... y {len(mismatched) - 10} más")
        lines.append("")
        lines.append(
            "El checkpoint NO es compatible con esta arquitectura. Comprueba que:"
        )
        lines.append(
            f"  - el archivo procede de un entrenamiento del mismo arch_name ({arch_name})"
        )
        lines.append("  - los OUT_CHANNELS del yaml coinciden con el modelo entrenado")
        lines.append(
            "  - no se ha entrenado con una resolución que cambien los stem strides"
        )
        raise RuntimeError("\n".join(lines))

    critical_missing = [
        k for k in missing_unexpected if k.startswith(critical_prefixes)
    ]
    if critical_missing:
        lines = [
            f"[{cls_name}] ABORT: faltan pesos CRÍTICOS en el checkpoint "
            f"(arch={arch_name}, ckpt={path}):"
        ]
        for k in critical_missing[:10]:
            lines.append(f"  {k}")
        if len(critical_missing) > 10:
            lines.append(f"  ... y {len(critical_missing) - 10} más")
        lines.append("")
        lines.append(
            "El checkpoint no contiene los pesos del backbone. Posibles causas:"
        )
        lines.append("  - el archivo no es del modelo esperado")
        lines.append(
            "  - el state_dict está bajo una clave no reconocida (no 'model' ni 'state_dict')"
        )
        raise RuntimeError("\n".join(lines))

    print(f"[{cls_name}] Cargados {len(matched)} tensores desde {path}")

    if missing_unexpected:
        print(
            f"[{cls_name}] AVISO: {len(missing_unexpected)} pesos del modelo no están "
            f"en el checkpoint (no estaban en expected_missing). "
            f"Ejemplos: {missing_unexpected[:5]}"
        )
    if unexpected:
        print(
            f"[{cls_name}] AVISO: {len(unexpected)} tensores del checkpoint no se usan "
            f"(típicamente head.*, norm.*, fc.*). Ejemplos: {unexpected[:5]}"
        )


@BACKBONE_REGISTRY.register()
class MetaformerBackbone(Backbone):
    """Adaptador de los modelos ``Metaformer`` a la interfaz de backbone
    de Detectron2. Soporta ``ARCH_NAME`` ∈ {``gated_cnn``,
    ``gated_cnn-dat``, ``gated_cnn-mamba``}.
    """

    def __init__(self, cfg: CN, input_shape: ShapeSpec):
        super().__init__()

        from models.factory import (
            _build_dat_backbone,
            _build_gcnn_backbone,
            _build_mamba_backbone,
        )
        from omegaconf import OmegaConf

        mf_cfg = cfg.MODEL.METAFORMER
        arch_name = mf_cfg.ARCH_NAME
        model_cfg = OmegaConf.create({"num_classes": 1000, "drop_path_rate": 0.1})

        if arch_name == "gated_cnn":
            metaformer = _build_gcnn_backbone(model_cfg)
        elif arch_name == "gated_cnn-dat":
            metaformer = _build_dat_backbone(model_cfg)
        elif arch_name == "gated_cnn-mamba":
            metaformer = _build_mamba_backbone(model_cfg)
        else:
            raise ValueError(
                f"MetaformerBackbone: ARCH_NAME='{arch_name}' no soportado. "
                f"Usa 'gated_cnn', 'gated_cnn-dat' o 'gated_cnn-mamba'. "
                f"Para ResNet50 usa la clase ResNet50Backbone con BACKBONE.NAME='ResNet50Backbone'."
            )

        self._out_features: List[str] = list(mf_cfg.OUT_FEATURES)
        self._out_channels: List[int] = list(mf_cfg.OUT_CHANNELS)
        self._out_strides: List[int] = list(mf_cfg.OUT_STRIDES)
        self._arch_name = arch_name

        assert (
            len(self._out_features) == len(self._out_channels) == len(self._out_strides)
        ), "OUT_FEATURES, OUT_CHANNELS y OUT_STRIDES deben tener la misma longitud"
        assert len(self._out_features) == len(metaformer.stages), (
            f"Tienes {len(metaformer.stages)} stages pero "
            f"{len(self._out_features)} OUT_FEATURES. Deben coincidir."
        )

        self.stem = metaformer.stem
        self.stages = metaformer.stages

        stage_dims = [s.in_dim for s in metaformer.stages]
        self.out_norms = nn.ModuleList([nn.LayerNorm(dim) for dim in stage_dims])

        weights_path: str = mf_cfg.WEIGHTS
        if weights_path:
            self._load_pretrained(weights_path)

    def _expected_missing_keys(self) -> Set[str]:
        expected = {"norm.weight", "norm.bias", "head.weight", "head.bias"}
        for i in range(len(self.out_norms)):
            expected.add(f"out_norms.{i}.weight")
            expected.add(f"out_norms.{i}.bias")
        return expected

    def _load_pretrained(self, path: str) -> None:
        load_pretrained_into(
            model=self,
            path=path,
            expected_missing=self._expected_missing_keys(),
            arch_name=self._arch_name,
            critical_prefixes=("stem", "stages"),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs: Dict[str, torch.Tensor] = {}
        x = self.stem(x)
        for i, stage in enumerate(self.stages):
            x, pre_down = stage(x, return_before_downsample=True)
            feat = self.out_norms[i](pre_down)
            feat = feat.permute(0, 3, 1, 2).contiguous()
            outputs[self._out_features[i]] = feat
        return outputs

    def output_shape(self) -> Dict[str, ShapeSpec]:
        return {
            name: ShapeSpec(channels=ch, stride=st)
            for name, ch, st in zip(
                self._out_features, self._out_channels, self._out_strides
            )
        }


@BACKBONE_REGISTRY.register()
class ResNet50Backbone(Backbone):
    """ResNet50 de torchvision adaptado a backbone de Detectron2 con
    carga de pesos desde un checkpoint de clasificación custom.

    Produce ``res2..res5`` en strides ``[4, 8, 16, 32]`` y canales
    ``[256, 512, 1024, 2048]``.

    Uso en yaml::

        MODEL:
          BACKBONE:
            NAME: "ResNet50Backbone"
          METAFORMER:
            WEIGHTS: "checkpoints/pretrain_resnet50/last.ckpt"
            OUT_FEATURES: ["res2", "res3", "res4", "res5"]
            OUT_CHANNELS: [256, 512, 1024, 2048]
            OUT_STRIDES:  [4, 8, 16, 32]
    """

    def __init__(self, cfg: CN, input_shape: ShapeSpec):
        super().__init__()

        from torchvision.models import resnet50

        self.model = resnet50(weights=None)
        self.model.fc = nn.Identity()
        self.model.avgpool = nn.Identity()

        mf_cfg = cfg.MODEL.METAFORMER
        self._out_features: List[str] = list(mf_cfg.OUT_FEATURES)
        self._out_channels: List[int] = list(mf_cfg.OUT_CHANNELS)
        self._out_strides: List[int] = list(mf_cfg.OUT_STRIDES)
        self._arch_name = "resnet50"

        assert self._out_features == ["res2", "res3", "res4", "res5"], (
            f"ResNet50Backbone: OUT_FEATURES debe ser ['res2','res3','res4','res5'], "
            f"recibido {self._out_features}"
        )
        assert self._out_channels == [256, 512, 1024, 2048], (
            f"ResNet50Backbone: OUT_CHANNELS debe ser [256,512,1024,2048], "
            f"recibido {self._out_channels}"
        )
        assert self._out_strides == [4, 8, 16, 32], (
            f"ResNet50Backbone: OUT_STRIDES debe ser [4,8,16,32], "
            f"recibido {self._out_strides}"
        )

        weights_path: str = mf_cfg.WEIGHTS
        if weights_path:
            self._load_pretrained(weights_path)

    def _expected_missing_keys(self) -> Set[str]:
        return {
            "fc.weight",
            "fc.bias",
            "model.fc.weight",
            "model.fc.bias",
        }

    def _load_pretrained(self, path: str) -> None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = checkpoint
        for key in ("model", "state_dict", "model_state_dict"):
            if isinstance(checkpoint, dict) and key in checkpoint:
                state_dict = checkpoint[key]
                break

        has_model_prefix = "model.conv1.weight" in state_dict
        has_unprefixed = "conv1.weight" in state_dict
        if not has_model_prefix and has_unprefixed:
            state_dict = {f"model.{k}": v for k, v in state_dict.items()}

        load_pretrained_into(
            model=self,
            path=path,
            state_dict=state_dict,
            expected_missing=self._expected_missing_keys(),
            arch_name=self._arch_name,
            critical_prefixes=("conv1", "bn1", "layer1", "layer2", "layer3", "layer4"),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        r2 = self.model.layer1(x)
        r3 = self.model.layer2(r2)
        r4 = self.model.layer3(r3)
        r5 = self.model.layer4(r4)

        return {
            self._out_features[0]: r2,
            self._out_features[1]: r3,
            self._out_features[2]: r4,
            self._out_features[3]: r5,
        }

    def output_shape(self) -> Dict[str, ShapeSpec]:
        return {
            name: ShapeSpec(channels=ch, stride=st)
            for name, ch, st in zip(
                self._out_features, self._out_channels, self._out_strides
            )
        }
