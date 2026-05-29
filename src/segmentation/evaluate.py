"""
evaluate.py
===========
Evalúa un checkpoint entrenado sobre COCO y genera una tabla comparativa
con los baselines de referencia (ResNet-50, Swin-T).

Uso:
    # Evaluación estándar
    python evaluate.py \
        --config-file configs/maskformer2_metaformer_coco_panoptic.yaml \
        --weights output/metaformer_tiny_coco_panoptic/model_final.pth

    # Con Test Time Augmentation (tarda más, sube ~1-2pp)
    python evaluate.py \
        --config-file configs/maskformer2_metaformer_coco_panoptic.yaml \
        --weights output/metaformer_tiny_coco_panoptic/model_final.pth \
        --tta

    # Evaluar sobre val2017 aunque el yaml diga otra cosa
    python evaluate.py \
        --config-file configs/maskformer2_metaformer_coco_panoptic.yaml \
        --weights output/metaformer_tiny_coco_panoptic/model_final.pth \
        DATASETS.TEST "('coco_2017_val_panoptic',)"
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.engine import default_setup
from detectron2.evaluation import (
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    SemSegEvaluator,
    inference_on_dataset,
)
from detectron2.modeling import build_model
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from mask2former import SemanticSegmentorWithTTA, add_maskformer2_config
from metaformer_backbone_d2 import MetaformerBackbone, add_metaformer_config  # noqa: F401

logger = logging.getLogger("detectron2")


# ---------------------------------------------------------------------------
# Baselines publicados para comparar en el TFG
# Fuente: paper Mask2Former (Cheng et al., 2022), val COCO 2017
# ---------------------------------------------------------------------------
BASELINES = {
    "ResNet-50": {
        "params_M": 44,
        "PQ": 51.9, "SQ": 83.0, "RQ": 61.9,
        "AP": 43.7,
        "mIoU": 57.8,
    },
    "ResNet-101": {
        "params_M": 63,
        "PQ": 52.6, "SQ": 83.2, "RQ": 62.6,
        "AP": 44.9,
        "mIoU": 59.5,
    },
    "Swin-T": {
        "params_M": 48,
        "PQ": 53.2, "SQ": 83.2, "RQ": 63.3,
        "AP": 45.7,
        "mIoU": 59.3,
    },
    "Swin-S": {
        "params_M": 69,
        "PQ": 54.5, "SQ": 83.8, "RQ": 64.7,
        "AP": 47.3,
        "mIoU": 61.7,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_evaluators(cfg, dataset_name: str, output_folder: str):
    evaluators = []
    meta = MetadataCatalog.get(dataset_name)

    if hasattr(meta, "panoptic_root"):
        evaluators.append(COCOPanopticEvaluator(dataset_name, output_folder))

    evaluators.append(COCOEvaluator(dataset_name, output_dir=output_folder))

    if hasattr(meta, "stuff_classes") and "with_sem_seg" in dataset_name:
        evaluators.append(
            SemSegEvaluator(dataset_name, distributed=False, output_dir=output_folder)
        )

    return DatasetEvaluators(evaluators)


def print_comparison_table(your_results: dict, your_params_M: float, model_name: str = "Tu Metaformer"):
    """Imprime tabla comparativa con baselines."""

    # Extraer métricas del dict de resultados de Detectron2
    pq = your_results.get("panoptic_seg", {}).get("PQ", None)
    sq = your_results.get("panoptic_seg", {}).get("SQ", None)
    rq = your_results.get("panoptic_seg", {}).get("RQ", None)
    ap = your_results.get("segm", {}).get("AP", None)
    miou = your_results.get("sem_seg", {}).get("mIoU", None)

    your_entry = {
        "params_M": round(your_params_M, 1),
        "PQ": round(pq, 1) if pq is not None else "—",
        "SQ": round(sq, 1) if sq is not None else "—",
        "RQ": round(rq, 1) if rq is not None else "—",
        "AP": round(ap, 1) if ap is not None else "—",
        "mIoU": round(miou, 1) if miou is not None else "—",
    }

    all_models = {**BASELINES, model_name: your_entry}

    # Cabecera
    sep = "─" * 82
    print(f"\n{sep}")
    print(f"  {'Backbone':<20} {'Params':>7}  {'PQ':>6}  {'SQ':>6}  {'RQ':>6}  {'AP':>6}  {'mIoU':>6}")
    print(sep)
    for name, m in all_models.items():
        marker = " ◄" if name == model_name else ""
        print(
            f"  {name:<20} {str(m['params_M'])+'M':>7}  "
            f"{str(m['PQ']):>6}  {str(m['SQ']):>6}  {str(m['RQ']):>6}  "
            f"{str(m['AP']):>6}  {str(m['mIoU']):>6}{marker}"
        )
    print(f"{sep}\n")

    # Delta respecto a ResNet-50
    baseline = BASELINES["ResNet-50"]
    print("  Δ respecto a ResNet-50 (referencia mínima del TFG):")
    for metric, key in [("PQ", "PQ"), ("AP", "AP"), ("mIoU", "mIoU")]:
        val = your_entry[key]
        if val != "—":
            delta = val - baseline[key]
            sign = "+" if delta >= 0 else ""
            print(f"    {metric}: {sign}{delta:.1f}")
    print()


def save_results(results: dict, output_dir: str, model_name: str):
    """Guarda resultados en JSON para el TFG."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(output_dir) / f"eval_results_{ts}.json"
    payload = {"model": model_name, "timestamp": ts, "results": results}
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Resultados guardados en {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def setup(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_metaformer_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # Sobreescribir weights si se pasa por argumento
    if args.weights:
        cfg.MODEL.WEIGHTS = args.weights

    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, name="evaluate")
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Evalúa MetaformerBackbone en COCO")
    parser.add_argument("--config-file", required=True, metavar="FILE")
    parser.add_argument("--weights", default="", metavar="FILE",
                        help="Ruta al checkpoint a evaluar")
    parser.add_argument("--tta", action="store_true",
                        help="Usar Test Time Augmentation")
    parser.add_argument("--model-name", default="Tu Metaformer",
                        help="Nombre para la tabla comparativa")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="Sobreescribir opciones del cfg (KEY VALUE ...)")

    # Truco para que default_setup no falle: simula args de Detectron2
    import sys
    args = parser.parse_args()
    args.num_gpus = 1
    args.num_machines = 1
    args.machine_rank = 0
    args.dist_url = "auto"
    args.resume = False
    args.eval_only = True

    cfg = setup(args)

    # Construir y cargar modelo
    model = build_model(cfg)
    model.eval()

    params_M = count_parameters(model) / 1e6
    logger.info(f"Parámetros entrenables: {params_M:.1f}M")

    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    logger.info(f"Checkpoint cargado: {cfg.MODEL.WEIGHTS}")

    # TTA opcional
    if args.tta:
        logger.info("Activando TTA...")
        model = SemanticSegmentorWithTTA(cfg, model)

    # Evaluar sobre cada dataset de test
    all_results = {}
    for dataset_name in cfg.DATASETS.TEST:
        logger.info(f"\nEvaluando sobre: {dataset_name}")
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        os.makedirs(output_folder, exist_ok=True)

        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = build_evaluators(cfg, dataset_name, output_folder)

        results = inference_on_dataset(model, data_loader, evaluator)
        all_results[dataset_name] = results

        logger.info(f"Resultados brutos ({dataset_name}): {results}")

    # Tabla comparativa
    first_results = next(iter(all_results.values()))
    print_comparison_table(first_results, params_M, model_name=args.model_name)

    # Guardar JSON
    save_results(all_results, cfg.OUTPUT_DIR, model_name=args.model_name)


if __name__ == "__main__":
    main()
