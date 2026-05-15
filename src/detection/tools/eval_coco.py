"""
eval_coco.py — Evaluación científica de métricas COCO para los modelos de detección.

Ejecuta evaluación sobre COCO val2017 y guarda resultados estructurados en JSON
y CSV para su inclusión en el TFG.

Requiere:
    - MMDetection >= 3.3.0 instalado
    - COCO val2017 descargado
    - Checkpoint de detección entrenado (Faster R-CNN completo)

Uso:
    # Evaluar una variante
    python src/detection/tools/eval_coco.py \\
        --config   configs/detection/faster_rcnn_gated_cnn_mamba_fpn_1x_coco.py \\
        --checkpoint checkpoints/detection/faster_rcnn_gated_cnn_mamba_1x.pth \\
        --out      outputs/detection/gated_cnn_mamba_eval.json

    # Comparar todas las variantes y generar tabla
    python src/detection/tools/eval_coco.py --compare \\
        --results-dir outputs/detection/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s"
)
log = logging.getLogger(__name__)

# Métricas COCO estándar
COCO_METRIC_NAMES = [
    "AP",  # mAP @ IoU=0.50:0.95
    "AP50",  # mAP @ IoU=0.50
    "AP75",  # mAP @ IoU=0.75
    "APs",  # AP objetos pequeños
    "APm",  # AP objetos medianos
    "APl",  # AP objetos grandes
    "AR1",  # AR con max 1 det/imagen
    "AR10",  # AR con max 10 det/imagen
    "AR100",  # AR con max 100 det/imagen
    "ARs",  # AR objetos pequeños
    "ARm",  # AR objetos medianos
    "ARl",  # AR objetos grandes
]


def _check_mmdet() -> None:
    try:
        import mmdet  # noqa: F401
        import mmengine  # noqa: F401
    except ImportError:
        log.error(
            "MMDetection no está instalado.\n"
            "Instalar con:\n"
            "  pip install mmdet mmcv mmengine\n"
            "  # o via mim:\n"
            "  pip install openmim && mim install mmdet"
        )
        sys.exit(1)


def run_eval(
    config_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    device: str = "cuda",
    show: bool = False,
) -> dict:
    """Evalúa un modelo Faster R-CNN sobre COCO val2017.

    Returns:
        dict con todas las métricas COCO.
    """
    _check_mmdet()

    from mmdet.apis import inference_detector, init_detector  # noqa: F401
    from mmengine.config import Config
    from mmengine.runner import Runner  # noqa: F401

    log.info(f"Cargando config: {config_path}")

    # Asegurar que el backbone custom está registrado ANTES de cargar la config
    _src = Path(__file__).resolve().parents[2]
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))
    import detection  # noqa: F401 — registra MetaFormerBackbone

    cfg = Config.fromfile(str(config_path))

    # Anular el preentrenamiento del backbone, ya que vamos a cargar el detector completo
    # Esto evita el error de "checkpoint file not found" al inicializar la arquitectura
    if hasattr(cfg, "model") and "backbone" in cfg.model:
        cfg.model.backbone.pretrained = None

    cfg.load_from = str(checkpoint_path)
    cfg.work_dir = str(output_path.parent)

    # Forzar evaluación (no entrenamiento)
    # Utilizamos el ann_file definido en la config para que no falle CocoMetric
    if (
        hasattr(cfg, "val_evaluator")
        and isinstance(cfg.val_evaluator, dict)
        and "ann_file" in cfg.val_evaluator
    ):
        ann_file = cfg.val_evaluator["ann_file"]
    else:
        ann_file = None

    cfg.test_evaluator = dict(
        type="CocoMetric", metric="bbox", classwise=True, ann_file=ann_file
    )

    log.info("Iniciando evaluación COCO...")
    runner = Runner.from_cfg(cfg)
    metrics = runner.test()

    # Estructurar resultados
    result = {
        "timestamp": datetime.now().isoformat(),
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "device": device,
        "metrics": metrics,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    log.info(f"Resultados guardados en: {output_path}")

    _print_metrics_table(metrics, label=config_path.stem)
    return result


def _print_metrics_table(metrics: dict, label: str = "") -> None:
    """Imprime tabla formateada de métricas COCO."""
    print(f"\n{'=' * 60}")
    print(f"  Resultados COCO: {label}")
    print(f"{'=' * 60}")
    print(f"  {'Métrica':<10}  {'Valor':>8}")
    print(f"  {'-' * 20}")
    for name in COCO_METRIC_NAMES:
        val = metrics.get(name, metrics.get(f"coco/{name}", -1))
        if isinstance(val, (int, float)):
            print(f"  {name:<10}  {val * 100:>7.2f}%")
    print(f"{'=' * 60}\n")


def compare_results(results_dir: Path) -> None:
    """Genera tabla comparativa a partir de JSON de evaluación guardados."""
    import csv

    json_files = list(results_dir.glob("*.json"))
    if not json_files:
        log.error(f"No se encontraron archivos JSON en {results_dir}")
        return

    rows = []
    for jf in sorted(json_files):
        with open(jf) as f:
            data = json.load(f)

        metrics = data.get("metrics", {})
        row = {"model": jf.stem}
        for name in COCO_METRIC_NAMES:
            val = metrics.get(name, metrics.get(f"coco/{name}", None))
            row[name] = round(float(val) * 100, 2) if val is not None else None
        rows.append(row)

    # Imprimir tabla
    col_w = 10
    header = f"{'Model':<40}" + "".join(f"{n:>{col_w}}" for n in COCO_METRIC_NAMES)
    print(f"\n{'=' * len(header)}")
    print(header)
    print("-" * len(header))
    for row in rows:
        vals = "".join(
            f"{row[n]:>{col_w}.2f}" if row[n] is not None else f"{'—':>{col_w}}"
            for n in COCO_METRIC_NAMES
        )
        print(f"{row['model']:<40}{vals}")
    print(f"{'=' * len(header)}\n")

    # Guardar CSV
    csv_path = results_dir / "comparison_table.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model"] + COCO_METRIC_NAMES)
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"Tabla comparativa guardada en: {csv_path}")

    # Generar LaTeX
    _generate_latex_table(rows, results_dir / "comparison_table.tex")


def _generate_latex_table(rows: list[dict], output_path: Path) -> None:
    """Genera tabla LaTeX lista para incluir en el TFG."""
    # Métricas principales para la tabla del TFG
    main_metrics = ["AP", "AP50", "AP75", "APs", "APm", "APl"]

    lines = [
        r"\begin{table}[ht]",
        r"  \centering",
        r"  \caption{Resultados de detección de objetos en COCO val2017 con Faster R-CNN (1$\times$ schedule).}",
        r"  \label{tab:detection_results}",
        r"  \begin{tabular}{l" + "c" * len(main_metrics) + "}",
        r"    \toprule",
        r"    \textbf{Backbone} & "
        + " & ".join(f"\\textbf{{{m}}}" for m in main_metrics)
        + r" \\",
        r"    \midrule",
    ]

    for row in rows:
        model_name = row["model"].replace("_", r"\_")
        vals = " & ".join(
            f"{row[m]:.1f}" if row[m] is not None else "—" for m in main_metrics
        )
        lines.append(f"    {model_name} & {vals} \\\\")

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]

    output_path.write_text("\n".join(lines) + "\n")
    log.info(f"Tabla LaTeX guardada en: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluación científica COCO para modelos MetaFormer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command")

    # Subcomando: evaluar un modelo
    eval_p = subparsers.add_parser("eval", help="Evaluar un modelo sobre COCO val2017")
    eval_p.add_argument("--config", "-c", type=Path, required=True)
    eval_p.add_argument("--checkpoint", "-k", type=Path, required=True)
    eval_p.add_argument(
        "--out", "-o", type=Path, default=Path("outputs/detection/eval_results.json")
    )
    eval_p.add_argument("--device", default="cuda")

    # Subcomando: comparar resultados guardados
    cmp_p = subparsers.add_parser("compare", help="Comparar JSONs de evaluación")
    cmp_p.add_argument(
        "--results-dir",
        type=Path,
        default=Path("outputs/detection"),
    )

    args = parser.parse_args()

    if args.command == "eval":
        run_eval(args.config, args.checkpoint, args.out, args.device)
    elif args.command == "compare":
        compare_results(args.results_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
