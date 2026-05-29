"""
visualize_detections.py — Visualiza predicciones del modelo entrenado sobre imágenes.

Requiere MMDetection instalado.

Uso:
    # Una sola imagen:
    python src/detection/tools/visualize_detections.py \\
        --config  src/configs/detection/faster_rcnn_gated_cnn_fpn_1x_coco.py \\
        --checkpoint outputs/detection/faster_rcnn_gated_cnn_fpn_1x/best.pth \\
        --input   /ruta/a/imagen.jpg \\
        --out-dir outputs/visualizations/gcnn

    # Directorio de imágenes:
    python src/detection/tools/visualize_detections.py \\
        --config  src/configs/detection/faster_rcnn_gated_cnn_fpn_1x_coco.py \\
        --checkpoint outputs/detection/faster_rcnn_gated_cnn_fpn_1x/best.pth \\
        --input   /data/coco/val2017/ \\
        --out-dir outputs/visualizations/gcnn \\
        --max-images 20 \\
        --score-thr 0.4

    # Modo interactivo (muestra en pantalla):
    python src/detection/tools/visualize_detections.py ... --show
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def _setup_path() -> None:
    _src = Path(__file__).resolve().parents[2]
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))


def _check_mmdet() -> None:
    try:
        import mmdet  # noqa: F401
    except ImportError:
        print(
            "[ERROR] MMDetection no está instalado.\n"
            "  pip install mmdet mmcv mmengine\n"
            "  o: pip install openmim && mim install mmdet"
        )
        sys.exit(1)


def collect_images(input_path: Path, max_images: int) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    imgs = sorted(
        p for p in input_path.rglob("*") if p.suffix.lower() in IMG_EXTENSIONS
    )
    return imgs[:max_images]


def load_model(config_path: Path, checkpoint_path: Path, device: str):
    """Carga el detector con MMDetection."""
    from mmdet.apis import init_detector

    _setup_path()
    import detection  # noqa: F401 — registra MetaFormerBackbone

    model = init_detector(str(config_path), str(checkpoint_path), device=device)
    return model


def run_inference(model, img_paths: list[Path], out_dir: Path,
                  score_thr: float, show: bool, wait_time: float) -> None:
    from mmdet.apis import inference_detector
    from mmdet.visualization import DetLocalVisualizer
    import mmcv

    out_dir.mkdir(parents=True, exist_ok=True)

    visualizer = DetLocalVisualizer(
        vis_backends=[dict(type="LocalVisBackend")],
        save_dir=str(out_dir),
    )
    # Registrar el dataset meta con las clases del modelo
    try:
        classes = model.dataset_meta.get("classes", None)
        if classes:
            visualizer.dataset_meta = model.dataset_meta
    except AttributeError:
        pass

    total = len(img_paths)
    print(f"\n  Procesando {total} imagen(s)  →  {out_dir}\n")
    class_counter: dict[str, int] = {}

    for i, img_path in enumerate(img_paths, 1):
        img = mmcv.imread(img_path, channel_order="rgb")
        result = inference_detector(model, img)

        # Estadísticas de clases detectadas
        pred = result.pred_instances
        scores = pred.scores.cpu().numpy()
        labels = pred.labels.cpu().numpy()
        keep = scores >= score_thr

        classes_detected = []
        if hasattr(model, "dataset_meta") and model.dataset_meta:
            cls_names = model.dataset_meta.get("classes", [])
            for lbl in labels[keep]:
                name = cls_names[lbl] if lbl < len(cls_names) else str(lbl)
                classes_detected.append(name)
                class_counter[name] = class_counter.get(name, 0) + 1

        n_det = int(keep.sum())
        print(f"  [{i:>4}/{total}] {img_path.name:<40}  {n_det} detecciones"
              f"  [{', '.join(set(classes_detected))}]")

        # Visualizar y guardar
        out_name = f"{img_path.stem}_det{img_path.suffix}"
        visualizer.add_datasample(
            out_name,
            img,
            data_sample=result,
            draw_gt=False,
            pred_score_thr=score_thr,
            show=show,
            wait_time=wait_time,
            out_file=str(out_dir / out_name),
        )

    # Resumen de clases
    if class_counter:
        print(f"\n  CLASES DETECTADAS (score ≥ {score_thr})")
        print(f"  {'─'*40}")
        for cls, cnt in sorted(class_counter.items(), key=lambda x: -x[1]):
            print(f"  {cls:<25} {cnt:>6} instancias")

    print(f"\n  Imágenes guardadas en: {out_dir}")


def main() -> None:
    _check_mmdet()

    p = argparse.ArgumentParser(
        description="Visualizar predicciones del detector MetaFormer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--config", "-c", required=True, type=Path,
                   help="Config MMDetection del modelo")
    p.add_argument("--checkpoint", "-k", required=True, type=Path,
                   help="Checkpoint del detector entrenado (.pth)")
    p.add_argument("--input", "-i", required=True, type=Path,
                   help="Imagen o directorio de imágenes")
    p.add_argument("--out-dir", "-o", type=Path,
                   default=Path("outputs/visualizations"),
                   help="Directorio de salida (default: outputs/visualizations)")
    p.add_argument("--score-thr", type=float, default=0.3,
                   help="Umbral de confianza mínimo (default: 0.3)")
    p.add_argument("--max-images", type=int, default=50,
                   help="Máximo de imágenes a procesar si input es directorio (default: 50)")
    p.add_argument("--device", default="cuda",
                   help="Dispositivo (default: cuda)")
    p.add_argument("--show", action="store_true",
                   help="Mostrar imágenes en pantalla (requiere display)")
    p.add_argument("--wait-time", type=float, default=0.0,
                   help="Segundos entre imágenes en modo --show (0 = esperar tecla)")
    args = p.parse_args()

    print("\n" + "═"*60)
    print("  Visualize Detections — MetaFormer Detection")
    print("═"*60)
    print(f"  Config    : {args.config}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Input     : {args.input}")
    print(f"  Score thr : {args.score_thr}")

    print("\n  Cargando modelo...", end="", flush=True)
    model = load_model(args.config, args.checkpoint, args.device)
    print(" OK")

    imgs = collect_images(args.input, args.max_images)
    if not imgs:
        print(f"[ERROR] No se encontraron imágenes en: {args.input}")
        sys.exit(1)
    print(f"  Imágenes encontradas: {len(imgs)}")

    run_inference(model, imgs, args.out_dir, args.score_thr, args.show, args.wait_time)
    print("\n" + "═"*60 + "\n")


if __name__ == "__main__":
    main()
