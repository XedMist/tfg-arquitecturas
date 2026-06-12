import argparse
import sys
from pathlib import Path
import mmcv
from mmengine.config import Config
from mmengine.registry import DATASETS
from mmdet.visualization import DetLocalVisualizer

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

def _setup_path() -> None:
    _src = Path(__file__).resolve().parents[2]
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

def collect_images(input_path: str) -> list[str]:
    path = Path(input_path)
    if path.is_file():
        return [path.name]
    elif path.is_dir():
        return [p.name for p in path.rglob("*") if p.suffix.lower() in IMG_EXTENSIONS]
    else:
        # Pudo ser solo el nombre del archivo parcial
        return [input_path]

def main():
    p = argparse.ArgumentParser(description="Visualizar Ground Truth de una o varias imágenes COCO")
    p.add_argument("--config", "-c", required=True, type=Path, help="Config MMDetection")
    p.add_argument("--input", "-i", required=True, type=str, help="Nombre de la imagen o directorio de imágenes")
    p.add_argument("--out-dir", "-o", type=Path, default=Path("outputs/visualizations/ground_truth"))
    args = p.parse_args()

    _setup_path()
    import detection  # noqa: F401

    cfg = Config.fromfile(args.config)
    
    # Construir el dataset de validación
    dataset = DATASETS.build(cfg.val_dataloader.dataset)
    
    img_names = collect_images(args.input)
    if not img_names:
        print(f"[ERROR] No se encontraron imágenes válidas a partir de '{args.input}'.")
        sys.exit(1)
        
    # Crear un diccionario para búsqueda rápida por nombre de archivo
    name_to_idx = {}
    for i in range(len(dataset)):
        info = dataset.get_data_info(i)
        name = Path(info['img_path']).name
        name_to_idx[name] = i

    # Preparar visualizador
    visualizer = DetLocalVisualizer(
        vis_backends=[dict(type="LocalVisBackend")],
        save_dir=str(args.out_dir),
    )
    visualizer.dataset_meta = dataset.metainfo
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    found = 0
    for img_name in img_names:
        idx = name_to_idx.get(img_name)
        if idx is None:
            # Intentar búsqueda parcial si no hay coincidencia exacta
            for name, i in name_to_idx.items():
                if img_name in name:
                    idx = i
                    break
                    
        if idx is None:
            print(f"[WARNING] No se encontró la imagen '{img_name}' en el dataset de validación. Saltando.")
            continue
            
        data = dataset[idx]
        img_path = data['data_samples'].img_path
        img = mmcv.imread(img_path, channel_order="rgb")
        
        out_name = f"{Path(img_path).stem}_gt.jpg"
        
        # Parche para MMDetection: extraer el tensor de bboxes
        if hasattr(data['data_samples'], 'gt_instances'):
            if hasattr(data['data_samples'].gt_instances, 'bboxes'):
                if hasattr(data['data_samples'].gt_instances.bboxes, 'tensor'):
                    data['data_samples'].gt_instances.bboxes = data['data_samples'].gt_instances.bboxes.tensor
                    
        # Dibujar Ground Truth
        visualizer.add_datasample(
            out_name,
            img,
            data_sample=data['data_samples'],
            draw_gt=True,
            draw_pred=False,
            show=False,
            out_file=str(args.out_dir / out_name)
        )
        print(f"  [{found+1}/{len(img_names)}] Ground Truth guardado en: {args.out_dir / out_name}")
        found += 1
        
    print(f"\nFinalizado. Se generaron {found} imágenes Ground Truth de {len(img_names)} solicitadas.")

if __name__ == "__main__":
    main()
