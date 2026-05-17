from typing import Dict, List, Optional

import torch
import torch.nn as nn

from detectron2.config import CfgNode as CN
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec

from models.metaformer import Metaformer, MetaFormerConfig


def add_metaformer_config(cfg: CN) -> None:
    cfg.MODEL.METAFORMER = CN()
    cfg.MODEL.METAFORMER.WEIGHTS = ""
    cfg.MODEL.METAFORMER.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.METAFORMER.OUT_CHANNELS = [96, 192, 384, 768]
    cfg.MODEL.METAFORMER.OUT_STRIDES = [4, 8, 16, 32]
    cfg.SOLVER.BACKBONE_LR_MULTIPLIER = 0.1


# ---------------------------------------------------------------------------
# El wrapper principal
# ---------------------------------------------------------------------------
@BACKBONE_REGISTRY.register()
class MetaformerBackbone(Backbone):
    def __init__(self, metaformer: Metaformer, cfg: CN):
        super().__init__()

        mf_cfg = cfg.MODEL.METAFORMER
        self._out_features: List[str] = mf_cfg.OUT_FEATURES
        self._out_channels: List[int] = mf_cfg.OUT_CHANNELS
        self._out_strides: List[int] = mf_cfg.OUT_STRIDES

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

    def _load_pretrained(self, path: str) -> None:
        checkpoint = torch.load(path, map_location="cpu")

        state_dict = checkpoint
        for key in ("model", "state_dict", "model_state_dict"):
            if key in checkpoint:
                state_dict = checkpoint[key]
                break

        # strict=False: ignora head y norm (que no existen aquí)
        # y carga todo lo demás (stem + stages)
        missing, unexpected = self.load_state_dict(state_dict, strict=False)

        # Filtramos los "missing" esperados (head, norm global)
        # para no alarmar al usuario con mensajes irrelevantes
        expected_missing = {"norm.weight", "norm.bias", "head.weight", "head.bias"}
        real_missing = [k for k in missing if k not in expected_missing]

        if real_missing:
            print(
                f"[MetaformerBackbone] Pesos NO cargados (inesperado): {real_missing}"
            )
        if unexpected:
            print(f"[MetaformerBackbone] Pesos inesperados en checkpoint: {unexpected}")

        print(f"[MetaformerBackbone] Pesos cargados desde {path}")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs: Dict[str, torch.Tensor] = {}

        # stem produce (B, H/4, W/4, C) en channels-last
        x = self.stem(x)

        for i, stage in enumerate(self.stages):
            # return_before_downsample=True devuelve (x_downsampled, x_pre_down)
            # x_pre_down tiene la resolución correcta para este nivel FPN
            x, pre_down = stage(x, return_before_downsample=True)

            # Normalización del feature map de salida
            feat = self.out_norms[i](pre_down)  # (B, H, W, C) channels-last

            # Detectron2 y Mask2Former esperan (B, C, H, W) — channels-first
            feat = feat.permute(0, 3, 1, 2).contiguous()

            name = self._out_features[i]
            outputs[name] = feat

        return outputs

    # ------------------------------------------------------------------
    # Información de forma — Detectron2 la usa para construir el
    # Pixel Decoder automáticamente
    # ------------------------------------------------------------------
    def output_shape(self) -> Dict[str, ShapeSpec]:
        return {
            name: ShapeSpec(channels=ch, stride=st)
            for name, ch, st in zip(
                self._out_features,
                self._out_channels,
                self._out_strides,
            )
        }


# ---------------------------------------------------------------------------
# Función de construcción — punto de entrada desde la config de Detectron2
# ---------------------------------------------------------------------------
def build_metaformer_backbone(cfg: CN, input_shape: ShapeSpec) -> MetaformerBackbone:
    """
    Esta función es la que Detectron2 llama internamente cuando ve
    MODEL.BACKBONE.NAME = "MetaformerBackbone".

    Construye el modelo Gated CNN (gcnn_2) basándose en la configuración de
    classification_gcnn_2.yaml y lo envuelve para Detectron2.
    """
    from omegaconf import OmegaConf
    from models.factory import _build_gcnn_backbone

    # Creamos la configuración del modelo alineada con classification_gcnn_2.yaml
    model_cfg = OmegaConf.create({"num_classes": 1000, "drop_path_rate": 0.1})

    # Construimos el backbone (gated_cnn)
    metaformer = _build_gcnn_backbone(model_cfg)

    return MetaformerBackbone(metaformer, cfg)


# Registramos build_metaformer_backbone como la función que construye
# el backbone cuando Detectron2 busca "MetaformerBackbone"
BACKBONE_REGISTRY.register()(build_metaformer_backbone)
