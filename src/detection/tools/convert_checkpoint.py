"""
convert_checkpoint.py — Convierte checkpoints de clasificación al formato
estándar MMDetection para ser usados como backbone preentrenado.

Los checkpoints de clasificación (generados por el trainer del proyecto) pueden
contener:
  - Estado del optimizador (de Lightning o trainer propio)
  - Prefijo 'model.' en las claves
  - Pesos de la cabeza de clasificación (head.*)
  - Pesos de la norma global (norm.*)

Este script extrae solo stem + stages y guarda en formato MMDetection:
  {'state_dict': {...}, 'meta': {'arch': ..., 'epoch': ...}}

Uso:
    python src/detection/tools/convert_checkpoint.py \\
        --input  checkpoints/backbone_classification/best.ckpt \\
        --output checkpoints/detection/gated_cnn_mamba_backbone.pth \\
        --arch   gated_cnn_mamba

    # Ver claves del checkpoint sin convertir
    python src/detection/tools/convert_checkpoint.py \\
        --input checkpoints/backbone_classification/best.ckpt \\
        --inspect
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s"
)
log = logging.getLogger(__name__)

# Prefijos que pertenecen al backbone (sin cabeza de clasificación)
_BACKBONE_PREFIXES = ("stem.", "stages.")

# Prefijos de Lightning que hay que quitar
_LIGHTNING_PREFIXES = ("model.", "_orig_mod.", "module.")


def _strip_prefix(key: str, prefixes: tuple[str, ...]) -> str:
    """Elimina un prefijo de una clave si lo tiene."""
    for p in prefixes:
        if key.startswith(p):
            return key[len(p):]
    return key


def _load_raw(path: Path) -> dict:
    """Carga un checkpoint raw de PyTorch/Lightning."""
    log.info(f"Cargando: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt


def _extract_state_dict(ckpt: dict) -> dict[str, torch.Tensor]:
    """Extrae el state_dict de un checkpoint (Lightning o raw)."""
    # Lightning guarda en 'state_dict'; otros en 'model_state_dict' o directamente
    for key in ("state_dict", "model_state_dict", "model"):
        if key in ckpt:
            log.info(f"  Usando clave '{key}' del checkpoint.")
            return ckpt[key]

    # Si el propio dict tiene tensores, es un state_dict directo
    if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        log.info("  Checkpoint parece ser un state_dict directo.")
        return ckpt

    raise ValueError(
        "No se encontró state_dict en el checkpoint. "
        f"Claves disponibles: {list(ckpt.keys())}"
    )


def _filter_backbone(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    Elimina prefijos de Lightning/DDP y filtra solo pesos de backbone.
    Convierte 'model.stem.conv1.weight' → 'stem.conv1.weight'
    """
    cleaned: dict[str, torch.Tensor] = {}
    skipped: list[str] = []

    for raw_key, tensor in state_dict.items():
        # Quitar prefijos de Lightning / DDP / torch.compile
        key = _strip_prefix(raw_key, _LIGHTNING_PREFIXES)
        # A veces hay doble prefijo: 'model.model.stem...'
        key = _strip_prefix(key, _LIGHTNING_PREFIXES)

        # Conservar solo stem y stages
        if any(key.startswith(p) for p in _BACKBONE_PREFIXES):
            cleaned[key] = tensor
        else:
            skipped.append(key)

    log.info(
        f"  Retenidos: {len(cleaned)} tensors | "
        f"Descartados: {len(skipped)} tensors"
    )
    if skipped:
        log.debug(f"  Descartados: {skipped[:10]} ...")

    return cleaned


def _inspect(path: Path) -> None:
    """Imprime todas las claves del checkpoint para diagnóstico."""
    ckpt = _load_raw(path)
    log.info(f"Claves de nivel superior: {list(ckpt.keys())}")

    try:
        sd = _extract_state_dict(ckpt)
    except ValueError as e:
        log.error(str(e))
        sys.exit(1)

    print(f"\nTotal parámetros: {len(sd)}")
    print("-" * 72)
    for i, (k, v) in enumerate(sd.items()):
        print(f"  {k:60s}  {str(tuple(v.shape)):20s}  {v.dtype}")
        if i >= 60:
            print(f"  ... ({len(sd) - 60} más)")
            break


def convert(
    input_path: Path,
    output_path: Path,
    arch: str,
    epoch: int | None = None,
) -> None:
    """Convierte y guarda el checkpoint en formato MMDetection."""
    ckpt = _load_raw(input_path)

    # Extraer epoch si está disponible
    if epoch is None:
        epoch = ckpt.get("epoch", -1)

    sd = _extract_state_dict(ckpt)
    backbone_sd = _filter_backbone(sd)

    if not backbone_sd:
        raise ValueError(
            "No se encontraron pesos de backbone. "
            "Revisa los prefijos con --inspect."
        )

    output = {
        "state_dict": backbone_sd,
        "meta": {
            "arch": arch,
            "epoch": epoch,
            "source": str(input_path),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    log.info(f"Checkpoint guardado en: {output_path}")
    log.info(f"  arch={arch}, epoch={epoch}, tensors={len(backbone_sd)}")

    # Verificar shapes clave
    shapes = {
        k: tuple(v.shape)
        for k, v in backbone_sd.items()
        if "conv1.weight" in k or "norm1.weight" in k or "norm2.weight" in k
    }
    if shapes:
        log.info("  Muestra de shapes:")
        for k, s in list(shapes.items())[:6]:
            log.info(f"    {k}: {s}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convierte checkpoints de clasificación a formato MMDetection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", "-i", required=True, type=Path,
        help="Path al checkpoint de clasificación (.ckpt o .pth)"
    )
    parser.add_argument(
        "--output", "-o", type=Path,
        default=None,
        help="Path de salida para el checkpoint convertido (.pth)"
    )
    parser.add_argument(
        "--arch", "-a",
        choices=["gated_cnn", "gated_cnn_mamba", "gated_cnn_dat"],
        default="gated_cnn_mamba",
        help="Variante de arquitectura (para metadata)"
    )
    parser.add_argument(
        "--inspect", action="store_true",
        help="Solo mostrar claves del checkpoint, sin convertir"
    )
    args = parser.parse_args()

    if args.inspect:
        _inspect(args.input)
        return

    if args.output is None:
        args.output = args.input.parent.parent / "detection" / f"{args.arch}_backbone.pth"

    convert(
        input_path=args.input,
        output_path=args.output,
        arch=args.arch,
    )


if __name__ == "__main__":
    main()
