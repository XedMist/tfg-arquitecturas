"""
test_backbone_shapes.py — Verificación rápida del backbone wrapper.

Comprueba que las shapes de salida son correctas para todos las variantes
sin necesidad de COCO ni MMDetection instalado.

Ejecutar desde el directorio raíz del proyecto:
    cd /home/xed/Clase/TFG/tfg-arquitecturas
    python src/detection/tools/test_backbone_shapes.py
"""

import sys
from pathlib import Path

# Asegurar que src/ está en el path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from detection.backbones import MetaFormerBackbone

# Shapes esperadas para input 224x224:
#   stride /4  → 56x56
#   stride /8  → 28x28
#   stride /16 → 14x14
#   stride /32 → 7x7
EXPECTED_SPATIAL = [56, 28, 14, 7]

ARCHS = {
    "gated_cnn": [96, 192, 384, 576],
    "gated_cnn_mamba": [96, 192, 384, 576],
    "gated_cnn_dat": [96, 192, 320, 512],
}


def test_arch(arch: str, expected_channels: list[int], device: str = "cpu") -> bool:
    print(f"\n{'=' * 60}")
    print(f"  Testeando: {arch}")
    print(f"{'=' * 60}")

    try:
        model = (
            MetaFormerBackbone(
                arch=arch,
                out_indices=(0, 1, 2, 3),
                frozen_stages=-1,  # nada congelado para el test
                pretrained=None,
            )
            .to(device)
            .eval()
        )

        x = torch.randn(1, 3, 224, 224, device=device)

        with torch.no_grad():
            outs = model(x)

        ok = True
        for i, (out, exp_c, exp_s) in enumerate(
            zip(outs, expected_channels, EXPECTED_SPATIAL)
        ):
            B, C, H, W = out.shape
            shape_ok = B == 1 and C == exp_c and H == exp_s and W == exp_s
            status = "✓" if shape_ok else "✗"
            print(
                f"  {status} C{i + 2}: {str(tuple(out.shape)):30s}  "
                f"esperado: (1, {exp_c}, {exp_s}, {exp_s})"
            )
            ok = ok and shape_ok

        # Contar params del backbone (sin head)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"\n  Parámetros backbone: {n_params:.1f}M")
        print(f"  out_channels: {model.out_channels}")
        return ok

    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDispositivo: {device}")
    print(f"PyTorch: {torch.__version__}")

    results = {}
    for arch, channels in ARCHS.items():
        results[arch] = test_arch(arch, channels, device)

    print(f"\n{'=' * 60}")
    print("  RESUMEN")
    print(f"{'=' * 60}")
    all_ok = True
    for arch, ok in results.items():
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {status}  {arch}")
        all_ok = all_ok and ok

    print()
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
