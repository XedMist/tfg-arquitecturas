import argparse
from pathlib import Path
import cv2
import numpy as np


def create_comparison_grid(base_dir: str, out_dir: str):
    base_path = Path(base_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Obtener todas las subcarpetas (que asumimos son los modelos y el ground truth)
    subdirs = [d for d in base_path.iterdir() if d.is_dir() and d.name != out_path.name]
    subdirs = sorted(subdirs)

    # Intentar poner el ground_truth primero si existe
    gt_dir = None
    for d in subdirs:
        if "ground_truth" in d.name.lower() or "gt" == d.name.lower():
            gt_dir = d
            break

    if gt_dir:
        subdirs.remove(gt_dir)
        subdirs.insert(0, gt_dir)

    if not subdirs:
        print(f"[ERROR] No se encontraron subcarpetas con imágenes en {base_dir}")
        return

    # Recopilar todos los nombres base de las imágenes
    all_images = set()
    for d in subdirs:
        for img_path in d.glob("*.*"):
            if img_path.suffix.lower() in [".jpg", ".png", ".jpeg"]:
                # Normalizar nombre eliminando sufijos añadidos por los scripts anteriores
                stem = img_path.stem.replace("_det", "").replace("_gt", "")
                all_images.add(stem)

    if not all_images:
        print(f"[ERROR] No se encontraron imágenes en las subcarpetas de {base_dir}.")
        return

    print(f"Carpetas detectadas (Columnas): {[d.name for d in subdirs]}")
    print(f"Total de imágenes únicas a comparar: {len(all_images)}")

    for stem in all_images:
        images_to_concat = []
        titles = []

        for d in subdirs:
            # Buscar la imagen en este subdirectorio
            matches = list(d.glob(f"{stem}*.*"))
            img = None
            if matches:
                img = cv2.imread(str(matches[0]))

            if img is not None:
                images_to_concat.append(img)
                titles.append(d.name)
            else:
                print(f"[Aviso] Falta la imagen '{stem}' en la carpeta '{d.name}'")

        if not images_to_concat:
            continue

        # Homogeneizar alturas para poder concatenar horizontalmente
        max_h = max(img.shape[0] for img in images_to_concat)

        # Espacio en blanco arriba para el texto
        header_h = 70
        font = cv2.FONT_HERSHEY_SIMPLEX

        processed_images = []
        for img, title in zip(images_to_concat, titles):
            h, w = img.shape[:2]
            if h != max_h:
                scale = max_h / h
                new_w = int(w * scale)
                img = cv2.resize(img, (new_w, max_h))
            else:
                new_w = w

            # Crear cabecera blanca
            header = np.ones((header_h, new_w, 3), dtype=np.uint8) * 255

            # Añadir texto centrado
            # Ajustar escala de texto basada en el ancho
            font_scale = min(1.5, new_w / 300)
            thickness = max(1, int(font_scale * 2))

            text_size = cv2.getTextSize(title, font, font_scale, thickness)[0]
            text_x = max(0, (new_w - text_size[0]) // 2)
            text_y = header_h - 20

            cv2.putText(
                header,
                title,
                (text_x, text_y),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA,
            )

            # Combinar cabecera con imagen
            combined = np.vstack((header, img))

            # Añadir un borde negro a la derecha para separar imágenes
            combined_with_border = cv2.copyMakeBorder(
                combined, 0, 0, 0, 5, cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
            processed_images.append(combined_with_border)

        # Crear un grid dinámico de 2 columnas (para hacer un formato 2x2, 3x2, etc.)
        cols = 2
        grid_rows = []
        for i in range(0, len(processed_images), cols):
            row_imgs = processed_images[i:i+cols]
            
            # Rellenar con imágenes en blanco si la fila no está completa
            while len(row_imgs) < cols:
                blank = np.ones_like(row_imgs[0]) * 255  # Fondo en blanco
                row_imgs.append(blank)
                
            row = np.hstack(row_imgs)
            grid_rows.append(row)

        # Asegurar que todas las filas tengan el mismo ancho antes de apilar verticalmente
        if grid_rows:
            target_w = max(r.shape[1] for r in grid_rows)
            for i in range(len(grid_rows)):
                if grid_rows[i].shape[1] != target_w:
                    grid_rows[i] = cv2.resize(grid_rows[i], (target_w, grid_rows[i].shape[0]))
            
            final_grid = np.vstack(grid_rows)
        else:
            # Fallback en caso de que algo falle (no debería)
            final_grid = processed_images[0] if processed_images else None

        out_file = out_path / f"{stem}_grid.jpg"
        cv2.imwrite(str(out_file), final_grid)
        print(f"Guardado: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Concatenar imágenes visualmente para comparar modelos."
    )
    parser.add_argument(
        "--base-dir",
        "-d",
        default="outputs/comparacion_visual",
        help="Directorio base que contiene las carpetas de los modelos (ej: outputs/comparacion_visual)",
    )
    parser.add_argument(
        "--out-dir",
        "-o",
        default="outputs/comparacion_visual/grids_comparativos",
        help="Directorio donde se guardarán las imágenes finales concatenadas",
    )

    args = parser.parse_args()
    create_comparison_grid(args.base_dir, args.out_dir)
