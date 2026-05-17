"""
infer.py
========
Corre inferencia sobre una imagen (o carpeta de imágenes) y guarda
la visualización de las máscaras predichas.

Uso:
    # Una imagen
    python infer.py \
        --config-file configs/maskformer2_metaformer_coco_panoptic.yaml \
        --weights output/metaformer_tiny_coco_panoptic/model_final.pth \
        --input foto.jpg \
        --output resultados/

    # Carpeta entera
    python infer.py \
        --config-file configs/maskformer2_metaformer_coco_panoptic.yaml \
        --weights output/metaformer_tiny_coco_panoptic/model_final.pth \
        --input imagenes/*.jpg \
        --output resultados/

    # Cambiar el umbral de confianza (por defecto 0.5)
    python infer.py ... --confidence-threshold 0.7

    # Modo panóptico / instancias / semántico
    python infer.py ... --task panoptic
    python infer.py ... --task instance
    python infer.py ... --task semantic
"""

import argparse
import os
import time
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode, Visualizer

from mask2former import add_maskformer2_config
from metaformer_backbone_d2 import MetaformerBackbone, add_metaformer_config  # noqa: F401

logger = setup_logger(name="infer")


# ---------------------------------------------------------------------------
# Predictor con threshold configurable
# ---------------------------------------------------------------------------
class MetaformerPredictor:
    """
    Wrapper sobre DefaultPredictor que:
      - Acepta imágenes en BGR (formato OpenCV) o rutas de archivo
      - Adapta el threshold de confianza
      - Mide el tiempo de inferencia
    """

    def __init__(self, cfg):
        self.cfg = cfg.clone()
        self.predictor = DefaultPredictor(cfg)

    def __call__(self, image_bgr: np.ndarray):
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = self.predictor(image_bgr)
        elapsed = time.perf_counter() - t0
        return outputs, elapsed


# ---------------------------------------------------------------------------
# Visualización
# ---------------------------------------------------------------------------
def visualize_panoptic(image_bgr, outputs, metadata, alpha=0.6):
    """
    Dibuja la segmentación panóptica: fondo con colores por clase,
    instancias con bordes y etiquetas.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    v = Visualizer(image_rgb, metadata=metadata, instance_mode=ColorMode.IMAGE)

    if "panoptic_seg" in outputs:
        panoptic_seg, segments_info = outputs["panoptic_seg"]
        vis = v.draw_panoptic_seg_predictions(
            panoptic_seg.to("cpu"), segments_info, alpha=alpha
        )
    elif "sem_seg" in outputs:
        sem_seg = outputs["sem_seg"].argmax(dim=0)
        vis = v.draw_sem_seg(sem_seg.to("cpu"), alpha=alpha)
    elif "instances" in outputs:
        instances = outputs["instances"].to("cpu")
        vis = v.draw_instance_predictions(instances)
    else:
        logger.warning("No se encontraron predicciones en la salida del modelo")
        return image_bgr

    result_rgb = vis.get_image()
    return cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)


def add_info_overlay(image, filename, elapsed_ms, num_segments=None):
    """Añade texto informativo sobre la imagen de resultado."""
    h, w = image.shape[:2]
    overlay = image.copy()

    # Fondo semitransparente para el texto
    cv2.rectangle(overlay, (0, h - 60), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)

    info = f"{Path(filename).name}  |  {elapsed_ms:.0f} ms"
    if num_segments:
        info += f"  |  {num_segments} segmentos"

    cv2.putText(
        image, info,
        (10, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
        (255, 255, 255), 1, cv2.LINE_AA
    )
    return image


def make_side_by_side(original_bgr, result_bgr):
    """Combina imagen original y resultado en una sola imagen."""
    h_orig, w_orig = original_bgr.shape[:2]
    h_res, w_res = result_bgr.shape[:2]

    # Igualar alturas
    if h_orig != h_res:
        result_bgr = cv2.resize(result_bgr, (w_res, h_orig))

    # Añadir etiquetas
    label_h = 30
    orig_label = np.zeros((label_h, w_orig, 3), dtype=np.uint8)
    res_label = np.zeros((label_h, w_res, 3), dtype=np.uint8)
    cv2.putText(orig_label, "Original", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(res_label, "Prediccion Mask2Former", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    orig_col = np.vstack([orig_label, original_bgr])
    res_col = np.vstack([res_label, result_bgr])

    # Separador vertical
    sep = np.ones((orig_col.shape[0], 3, 3), dtype=np.uint8) * 80
    return np.hstack([orig_col, sep, res_col])


# ---------------------------------------------------------------------------
# Procesamiento por imagen
# ---------------------------------------------------------------------------
def process_image(predictor, image_path, metadata, output_dir, task, alpha=0.6):
    image_bgr = read_image(image_path, format="BGR")
    original_bgr = image_bgr.copy()

    outputs, elapsed = predictor(image_bgr)
    elapsed_ms = elapsed * 1000

    # Contar segmentos detectados
    num_segments = None
    if "panoptic_seg" in outputs:
        _, segments_info = outputs["panoptic_seg"]
        num_segments = len(segments_info)
    elif "instances" in outputs:
        num_segments = len(outputs["instances"])

    logger.info(
        f"{Path(image_path).name}: {elapsed_ms:.0f} ms"
        + (f", {num_segments} segmentos" if num_segments else "")
    )

    # Visualizar
    result_bgr = visualize_panoptic(image_bgr, outputs, metadata, alpha=alpha)
    result_bgr = add_info_overlay(result_bgr, image_path, elapsed_ms, num_segments)

    # Imagen lado a lado
    combined = make_side_by_side(original_bgr, result_bgr)

    # Guardar
    stem = Path(image_path).stem
    out_path = Path(output_dir) / f"{stem}_pred.jpg"
    combined_path = Path(output_dir) / f"{stem}_comparison.jpg"

    cv2.imwrite(str(out_path), result_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(combined_path), combined, [cv2.IMWRITE_JPEG_QUALITY, 95])

    logger.info(f"  Guardado: {out_path}")
    logger.info(f"  Comparación: {combined_path}")

    return {
        "file": image_path,
        "elapsed_ms": round(elapsed_ms, 1),
        "num_segments": num_segments,
        "output": str(out_path),
    }


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def setup(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_metaformer_config(cfg)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    if args.weights:
        cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Inferencia visual con Mask2Former + Metaformer")
    parser.add_argument("--config-file", required=True, metavar="FILE")
    parser.add_argument("--weights", default="", metavar="FILE",
                        help="Ruta al checkpoint")
    parser.add_argument("--input", nargs="+", required=True,
                        help="Imágenes de entrada (rutas o glob)")
    parser.add_argument("--output", required=True,
                        help="Directorio de salida")
    parser.add_argument("--task", default="panoptic",
                        choices=["panoptic", "instance", "semantic"],
                        help="Tipo de segmentación a visualizar")
    parser.add_argument("--confidence-threshold", type=float, default=0.5,
                        help="Umbral de confianza para instancias (0-1)")
    parser.add_argument("--alpha", type=float, default=0.6,
                        help="Opacidad de las máscaras (0=transparente, 1=sólido)")
    parser.add_argument("--dataset", default="coco_2017_val_panoptic",
                        help="Dataset del que usar los metadatos (nombres de clases)")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()

    # Setup mínimo para que default_setup no falle
    args.num_gpus = 1
    args.num_machines = 1
    args.machine_rank = 0
    args.dist_url = "auto"
    args.resume = False
    args.eval_only = True

    cfg = setup(args)
    os.makedirs(args.output, exist_ok=True)

    # Metadatos COCO (nombres de clases y colores)
    metadata = MetadataCatalog.get(args.dataset)

    # Predictor
    predictor = MetaformerPredictor(cfg)
    logger.info(f"Modelo cargado desde: {cfg.MODEL.WEIGHTS}")
    logger.info(f"Dispositivo: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # Resolver lista de imágenes
    image_paths = []
    for pattern in args.input:
        paths = glob(pattern)
        if paths:
            image_paths.extend(sorted(paths))
        elif os.path.isfile(pattern):
            image_paths.append(pattern)
        else:
            logger.warning(f"No se encontraron imágenes con: {pattern}")

    if not image_paths:
        logger.error("No hay imágenes que procesar.")
        return

    logger.info(f"Procesando {len(image_paths)} imagen(es)...")

    # Procesar
    summaries = []
    for i, path in enumerate(image_paths, 1):
        logger.info(f"[{i}/{len(image_paths)}] {path}")
        try:
            summary = process_image(
                predictor, path, metadata, args.output,
                task=args.task, alpha=args.alpha
            )
            summaries.append(summary)
        except Exception as e:
            logger.error(f"  Error procesando {path}: {e}")

    # Resumen final
    if summaries:
        times = [s["elapsed_ms"] for s in summaries]
        logger.info(
            f"\nResumen: {len(summaries)} imágenes procesadas  |  "
            f"Tiempo medio: {np.mean(times):.0f} ms  |  "
            f"FPS medio: {1000/np.mean(times):.1f}"
        )
        logger.info(f"Resultados guardados en: {args.output}/")


if __name__ == "__main__":
    main()
