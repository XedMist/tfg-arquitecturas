"""
data_loading.py
---------------
Dataset y DataLoaders para ImageNet-1K (y cualquier dataset en formato ImageFolder)
con Albumentations como backend de augmentación.

Estructura esperada en disco:
    {root}/
      train/
        images/
          {class_name}/
            *.jpg | *.jpeg | *.png | *.webp
      val/
        images/
          {class_name}/
            ...
      test/          (opcional, sin uso en entrenamiento)
        images/
          {class_name}/
            ...

Subset estratificado (uso parcial del dataset):
    Se puede entrenar con una fracción del dataset manteniendo el balance
    entre clases mediante muestreo aleatorio estratificado con seed fijo.
    Esto garantiza reproducibilidad total y es la práctica estándar en
    trabajos que reportan resultados con subsets de ImageNet
    (Yalniz et al., 2019; Touvron et al., 2021).
    El parámetro ``subset_fraction`` (float en (0, 1]) se aplica
    independientemente a train y val, y se controla desde config:
        cfg.data.subset_fraction: 0.25   # 25 % por clase
        cfg.data.subset_seed: 42         # seed reproducible

Transformaciones de entrenamiento (canónicas según literatura):
    - RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3/4, 4/3))
        → He et al. (2016), Szegedy et al. (2015), torchvision baseline
    - HorizontalFlip(p=0.5)
        → Estándar universal en clasificación ImageNet
    - ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        → He et al. (2016); valores usados también en SimCLR (Chen et al., 2020)
    - GaussianBlur(p=0.1)
        → Presente en He et al. (2019) bag-of-tricks; recomendado en torchvision
          aunque con p baja para supervised learning (≠ SSL donde p≈0.5)
    - Grayscale(p=0.02)
        → Regularización leve; presente en SimCLR y variantes
    - Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

Transformaciones de validación (estándar):
    - SmallestMaxSize(256) + CenterCrop(224)
        → Evaluación canónica ImageNet desde Krizhevsky et al. (2012)
    - Normalize

Referencias:
    He et al. (2016). Deep Residual Learning for Image Recognition. CVPR.
    He et al. (2019). Bag of Tricks for Image Classification. CVPR.
    Chen et al. (2020). A Simple Framework for Contrastive Learning. ICML.
    Yalniz et al. (2019). Billion-scale semi-supervised learning. arXiv.
    Touvron et al. (2021). Training data-efficient image transformers. ICML.
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Constantes ImageNet (medias y desviaciones del conjunto de entrenamiento)
# ---------------------------------------------------------------------------

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class ImageFolderAlbu(Dataset):
    """
    Dataset en formato ImageFolder compatible con Albumentations.

    Soporta la estructura con subdirectorio 'images/' intermedio:
        root/{images/}{class_name}/{image}.ext

    El subdirectorio 'images/' se detecta automáticamente: si existe,
    se desciende a él antes de buscar las carpetas de clase.

    Parameters
    ----------
    root : str
        Ruta a la raíz del split (p.ej. ``/data/imagenet/train``).
    transform : Callable, optional
        Pipeline de Albumentations (debe terminar en ``ToTensorV2``).
    """

    # Extensiones aceptadas (en minúscula).
    _VALID_EXT: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp"})

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        subset_fraction: float = 1.0,
        subset_seed: int = 42,
    ) -> None:
        """
        Parameters
        ----------
        root : str
            Ruta a la raíz del split (p.ej. ``/data/imagenet/train``).
        transform : Callable, optional
            Pipeline de Albumentations (debe terminar en ``ToTensorV2``).
        subset_fraction : float
            Fracción del dataset a usar, en (0, 1]. Por defecto 1.0 (todo).
            El muestreo es estratificado: se selecciona la misma proporción
            en cada clase, garantizando balance. Con subset_fraction=0.25
            se conservan exactamente ceil(N_k * 0.25) muestras de cada
            clase k, donde N_k es el tamaño original de esa clase.
        subset_seed : int
            Semilla para el generador aleatorio. Fija la partición de forma
            reproducible entre ejecuciones. Por defecto 42.
        """
        if not (0.0 < subset_fraction <= 1.0):
            raise ValueError(
                f"subset_fraction debe estar en (0, 1], got {subset_fraction}"
            )
        self.root = Path(root)
        self.transform = transform
        self.subset_fraction = subset_fraction
        self.subset_seed = subset_seed
        self.samples: list[tuple[str, int]] = []
        self.classes: list[str] = []
        self._load_samples()

    # ------------------------------------------------------------------
    # Carga de muestras
    # ------------------------------------------------------------------

    def _resolve_class_root(self) -> Path:
        """
        Devuelve la carpeta que contiene las subcarpetas de clase.

        Si existe ``self.root/images/`` se considera que es la carpeta
        con las clases; en caso contrario se usa ``self.root`` directamente.
        Este comportamiento cubre tanto la estructura de ImageNet-1K en
        clústeres (con el nivel 'images/') como la estructura plana.
        """
        candidate = self.root / "images"
        if candidate.is_dir():
            return candidate
        return self.root

    def _load_samples(self) -> None:
        class_root = self._resolve_class_root()
        class_dirs = sorted([d for d in class_root.iterdir() if d.is_dir()])

        if not class_dirs:
            raise RuntimeError(
                f"No se encontraron subcarpetas de clase en '{class_root}'. "
                "Comprueba que la ruta es correcta y que la estructura es "
                "{split}/images/{class}/{image}.ext"
            )

        self.classes = [d.name for d in class_dirs]
        class_to_idx: dict[str, int] = {cls: i for i, cls in enumerate(self.classes)}

        # Agrupar todas las muestras por clase antes de muestrear,
        # para poder aplicar el subset de forma estratificada.
        samples_per_class: dict[int, list[tuple[str, int]]] = defaultdict(list)
        for cls_dir in class_dirs:
            cls_idx = class_to_idx[cls_dir.name]
            for img_path in cls_dir.glob("*"):
                if img_path.suffix.lower() in self._VALID_EXT:
                    samples_per_class[cls_idx].append((str(img_path), cls_idx))

        if not any(samples_per_class.values()):
            raise RuntimeError(
                f"No se encontraron imágenes válidas bajo '{class_root}'. "
                f"Extensiones aceptadas: {self._VALID_EXT}"
            )

        # Muestreo estratificado: se aplica la misma fracción por clase,
        # usando un generador con seed fijo para reproducibilidad total.
        # Se usa math.ceil para garantizar al menos 1 muestra por clase
        # incluso con fracciones muy pequeñas.
        import math

        rng = random.Random(self.subset_seed)

        all_samples: list[tuple[str, int]] = []
        for cls_idx in sorted(samples_per_class.keys()):
            cls_samples = samples_per_class[cls_idx]
            # Orden determinista antes de muestrear (evita dependencia del
            # orden de glob, que puede variar entre sistemas de ficheros).
            cls_samples.sort(key=lambda x: x[0])
            n_keep = math.ceil(len(cls_samples) * self.subset_fraction)
            chosen = rng.sample(cls_samples, n_keep)
            all_samples.extend(chosen)

        self.samples = all_samples

    # ------------------------------------------------------------------
    # Interfaz Dataset
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]

        # cv2 lee BGR → convertir a RGB antes de cualquier augmentación.
        image = cv2.imread(path)
        if image is None:
            raise OSError(f"No se pudo leer la imagen: '{path}'")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, label


# ---------------------------------------------------------------------------
# Pipelines de augmentación
# ---------------------------------------------------------------------------


def build_train_transform(cfg: DictConfig) -> A.Compose:
    """
    Pipeline de augmentación para entrenamiento en ImageNet.

    Implementa el protocolo canónico de la literatura de clasificación
    supervisada (He et al., 2016; He et al., 2019):

    1. RandomResizedCrop  — escala y relación de aspecto aleatorias.
    2. HorizontalFlip     — flip horizontal con p=0.5.
    3. ColorJitter        — perturbación de brillo/contraste/saturación/hue.
    4. GaussianBlur       — suavizado leve (p=0.1); reduce sobreajuste a
                            detalles de alta frecuencia (He et al., 2019).
    5. ToGray             — conversión a escala de grises con p baja (0.02)
                            como regularización adicional.
    6. Normalize          — media y std de ImageNet-1K.
    7. ToTensorV2         — ndarray HWC → tensor CHW sin copia de datos.

    Parameters
    ----------
    cfg : DictConfig
        Configuración Hydra. Se esperan las claves:
            cfg.augmentation.train.{random_resized_crop, horizontal_flip_p,
                color_jitter.{brightness,contrast,saturation,hue,p},
                grayscale_p, gaussian_blur_p}
            cfg.data.{mean, std}
    """
    aug = cfg.augmentation.train
    data = cfg.data

    crop_size: int = aug.random_resized_crop  # típicamente 224

    transforms = [
        # 1. RandomResizedCrop: escala uniforme en [0.08, 1.0] y ratio en
        #    [3/4, 4/3] siguiendo He et al. (2016) y torchvision defaults.
        A.RandomResizedCrop(
            size=(crop_size, crop_size),
            scale=(0.08, 1.0),
            ratio=(3.0 / 4.0, 4.0 / 3.0),
            interpolation=cv2.INTER_LINEAR,
        ),
        # 2. Flip horizontal (p=0.5 es invariante de clase en ImageNet).
        A.HorizontalFlip(p=aug.horizontal_flip_p),
        # 3. ColorJitter: perturbación de color. Valores por defecto
        #    (brightness=contrast=saturation=0.4, hue=0.1) son los de
        #    He et al. (2016) y SimCLR (Chen et al., 2020).
        A.ColorJitter(
            brightness=aug.color_jitter.brightness,
            contrast=aug.color_jitter.contrast,
            saturation=aug.color_jitter.saturation,
            hue=aug.color_jitter.hue,
            p=aug.color_jitter.p,
        ),
        # 4. GaussianBlur (AÑADIDO): recomendado en He et al. (2019) bag-of-tricks.
        #    p=0.1 para supervised learning (en SSL como SimCLR se usa p≈0.5).
        #    sigma_limit empírico [0.1, 2.0] cubre desde suavizado mínimo
        #    hasta el equivalente a un kernel 5×5.
        A.GaussianBlur(
            blur_limit=(3, 7),
            sigma_limit=(0.1, 2.0),
            p=aug.get("gaussian_blur_p", 0.1),
        ),
        # 5. ToGray: regularización leve.
        A.ToGray(p=aug.grayscale_p),
        # 6. Normalización con media y std de ImageNet-1K.
        A.Normalize(mean=list(data.mean), std=list(data.std)),
        # 7. Conversión a tensor PyTorch (CHW, float32).
        ToTensorV2(),
    ]

    return A.Compose(transforms)


def build_val_transform(cfg: DictConfig) -> A.Compose:
    """
    Pipeline de evaluación para ImageNet.

    Protocolo estándar desde Krizhevsky et al. (2012):
        1. SmallestMaxSize(256): redimensiona la dimensión más corta a 256
           manteniendo la relación de aspecto.
        2. CenterCrop(224): recorte central de 224×224.
        3. Normalize + ToTensorV2.

    No se aplica ninguna augmentación estocástica en validación/test.

    Parameters
    ----------
    cfg : DictConfig
        cfg.augmentation.val.{resize, center_crop}
        cfg.data.{mean, std}
    """
    aug = cfg.augmentation.val
    data = cfg.data

    return A.Compose(
        [
            # Redimensionar la dimensión más corta preservando aspecto.
            A.SmallestMaxSize(
                max_size=aug.resize,
                interpolation=cv2.INTER_LINEAR,
            ),
            # Recorte central: elimina bordes preservando el contenido central.
            A.CenterCrop(size=(aug.center_crop, aug.center_crop)),
            A.Normalize(mean=list(data.mean), std=list(data.std)),
            ToTensorV2(),
        ]
    )


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------


def build_classification_loaders(cfg: DictConfig) -> tuple[DataLoader, DataLoader]:
    """
    Construye DataLoaders para entrenamiento y validación.

    Subset estratificado:
        Si ``cfg.data.subset_fraction < 1.0``, se aplica muestreo aleatorio
        estratificado con ``cfg.data.subset_seed`` como semilla. El subset
        se aplica tanto a train como a val, de forma que la evaluación
        también sea sobre una partición representativa y reproducible.
        Esta práctica es estándar en trabajos que reportan resultados con
        subsets de ImageNet (Touvron et al., 2021; Yalniz et al., 2019).

    Convenciones de eficiencia aplicadas:
        - ``drop_last=True`` en train: evita batches pequeños al final de cada
          época que pueden producir gradientes ruidosos (especialmente con BN).
        - ``pin_memory=True``: transfiere tensores a memoria fijada (page-locked)
          para aceleración de H2D transfers.
        - ``persistent_workers=True``: mantiene los procesos worker vivos entre
          épocas, eliminando el overhead de fork/join por época.
        - ``prefetch_factor=2`` (default): cada worker pre-carga 2 batches en
          background; valor conservador que evita saturar RAM.
        - El batch_size de validación se duplica ya que no hay paso de backward
          ni almacenamiento de activaciones intermedias.

    Parameters
    ----------
    cfg : DictConfig
        cfg.data.{root, workers, pin_memory, prefetch_factor,
                  subset_fraction, subset_seed}
        cfg.training.batch_size

    Returns
    -------
    train_loader, val_loader : tuple[DataLoader, DataLoader]
    """
    root = Path(cfg.data.root)

    # subset_fraction=1.0 y subset_seed=42 como defaults seguros:
    # si no se especifican en config se usa el dataset completo.
    subset_fraction: float = cfg.data.get("subset_fraction", 1.0)
    subset_seed: int = cfg.data.get("subset_seed", 42)

    train_ds = ImageFolderAlbu(
        root=str(root / "train"),
        transform=build_train_transform(cfg),
        subset_fraction=subset_fraction,
        subset_seed=subset_seed,
    )
    val_ds = ImageFolderAlbu(
        root=str(root / "val"),
        transform=build_val_transform(cfg),
        subset_fraction=subset_fraction,
        subset_seed=subset_seed,
    )

    num_workers: int = cfg.data.workers
    # prefetch_factor solo tiene efecto cuando num_workers > 0;
    # con workers=0 la carga es síncrona y el argumento se ignora
    # (o lanza error en versiones antiguas de PyTorch).
    prefetch_factor: Optional[int] = (
        cfg.data.get("prefetch_factor", 2) if num_workers > 0 else None
    )
    persistent_workers: bool = num_workers > 0

    common_kwargs: dict = dict(
        num_workers=num_workers,
        pin_memory=cfg.data.pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        drop_last=True,  # evita batch final pequeño con BN
        **common_kwargs,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size * 2,  # sin backward → cabe el doble
        shuffle=False,
        drop_last=False,
        **common_kwargs,
    )

    return train_loader, val_loader
