from __future__ import annotations

import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional

import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


class ImageFolderAlbu(Dataset):
    """ImageFolder-compatible dataset backed by Albumentations transforms.

    Expects the following directory layout (torchvision ImageFolder style)::

        root/
          {class_a}/image1.jpg
          {class_a}/image2.png
          {class_b}/image1.jpg
          ...

    or with an optional ``images/`` sub-directory (common in some datasets)::

        root/images/{class_a}/...

    Args:
        root:             Path to the split directory (e.g. ``data/train``).
        transform:        Albumentations ``Compose`` pipeline.
        subset_fraction:  Fraction of images to keep per class (1.0 = all).
                          Stratified sampling preserves class balance.
        subset_seed:      RNG seed for reproducible sub-sampling.
    """

    _VALID_EXT: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp"})

    def __init__(
        self,
        root: str | Path,
        transform: Optional[Callable] = None,
        subset_fraction: float = 1.0,
        subset_seed: int = 42,
    ) -> None:
        if not (0.0 < subset_fraction <= 1.0):
            raise ValueError(
                f"subset_fraction must be in (0, 1], got {subset_fraction}"
            )

        self.root = Path(root)
        self.transform = transform
        self.subset_fraction = subset_fraction
        self.subset_seed = subset_seed
        self.samples: list[tuple[str, int]] = []
        self.classes: list[str] = []

        self._load_samples()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_class_root(self) -> Path:
        """Support both ``root/{cls}/`` and ``root/images/{cls}/``."""
        candidate = self.root / "images"
        return candidate if candidate.is_dir() else self.root

    def _load_samples(self) -> None:
        class_root = self._resolve_class_root()
        class_dirs = sorted(d for d in class_root.iterdir() if d.is_dir())

        if not class_dirs:
            raise RuntimeError(
                f"No class sub-directories found under '{class_root}'. "
                "Expected layout: {root}/{class}/{image}.ext"
            )

        self.classes = [d.name for d in class_dirs]
        class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        samples_per_class: dict[int, list[tuple[str, int]]] = defaultdict(list)
        for cls_dir in class_dirs:
            idx = class_to_idx[cls_dir.name]
            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() in self._VALID_EXT:
                    samples_per_class[idx].append((str(img_path), idx))

        if not any(samples_per_class.values()):
            raise RuntimeError(
                f"No valid images found under '{class_root}'. "
                f"Accepted extensions: {self._VALID_EXT}"
            )

        rng = random.Random(self.subset_seed)
        all_samples: list[tuple[str, int]] = []

        for cls_idx in sorted(samples_per_class.keys()):
            cls_samples = sorted(samples_per_class[cls_idx], key=lambda x: x[0])
            n_keep = max(1, math.ceil(len(cls_samples) * self.subset_fraction))
            all_samples.extend(rng.sample(cls_samples, n_keep))

        self.samples = all_samples

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]

        image = cv2.imread(path)
        if image is None:
            raise OSError(f"Could not read image: '{path}'")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        return image, label


# ---------------------------------------------------------------------------
# Transform factories
# ---------------------------------------------------------------------------


def build_train_transform(cfg: DictConfig) -> A.Compose:
    """DeiT / Swin training augmentation pipeline.

    ColorJitter parameters are interpreted as *absolute* half-ranges
    (Albumentations convention) so we convert from the [0, 1] DeiT values:
    jitter_factor → limit = jitter_factor (Albu adds ± symmetrically).
    """
    aug = cfg.augmentation.train
    data = cfg.data

    crop_size: int = int(aug.random_resized_crop)

    transforms: list[A.BasicTransform] = [
        A.RandomResizedCrop(
            size=(crop_size, crop_size),
            scale=(0.08, 1.0),  # standard ImageNet crop range
            ratio=(3.0 / 4.0, 4.0 / 3.0),
            interpolation=cv2.INTER_LINEAR,
        ),
        A.HorizontalFlip(p=float(aug.horizontal_flip_p)),
        A.ColorJitter(
            brightness=float(aug.color_jitter.brightness),
            contrast=float(aug.color_jitter.contrast),
            saturation=float(aug.color_jitter.saturation),
            hue=float(aug.color_jitter.hue),
            p=float(aug.color_jitter.p),
        ),
    ]

    # Optional Gaussian blur (DeiT-III adds this at p=0.1).
    gaussian_blur_p: float = float(aug.get("gaussian_blur_p", 0.0))
    if gaussian_blur_p > 0.0:
        transforms.append(
            A.GaussianBlur(
                blur_limit=(3, 7),
                sigma_limit=(0.1, 2.0),
                p=gaussian_blur_p,
            )
        )

    # Random grayscale (DeiT uses p=0.2).
    grayscale_p: float = float(aug.get("grayscale_p", 0.0))
    if grayscale_p > 0.0:
        transforms.append(A.ToGray(p=grayscale_p))

    transforms += [
        A.Normalize(mean=list(data.mean), std=list(data.std)),
        ToTensorV2(),
    ]

    return A.Compose(transforms)


def build_val_transform(cfg: DictConfig) -> A.Compose:
    """Standard ImageNet validation pipeline.

    Resize the shortest side to ``resize`` (typically 256 for 224-crop
    models), then centre-crop to ``center_crop`` (typically 224).
    This matches the torchvision / timm evaluation convention.
    """
    aug = cfg.augmentation.val
    data = cfg.data

    resize_size: int = int(aug.resize)
    center_crop_size: int = int(aug.center_crop)

    return A.Compose(
        [
            A.SmallestMaxSize(
                max_size=resize_size,
                interpolation=cv2.INTER_LINEAR,
            ),
            A.CenterCrop(height=center_crop_size, width=center_crop_size),
            A.Normalize(mean=list(data.mean), std=list(data.std)),
            ToTensorV2(),
        ]
    )


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------


def build_classification_loaders(
    cfg: DictConfig,
) -> tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders.

    Validation batch size is doubled vs. training (no gradients, so we can
    use larger batches to speed up evaluation) — standard timm practice.
    """
    root = Path(cfg.data.root)
    subset_fraction = float(cfg.data.get("subset_fraction", 1.0))
    subset_seed = int(cfg.data.get("subset_seed", 42))
    num_workers = int(cfg.data.workers)
    pin_memory = bool(cfg.data.pin_memory)
    prefetch_factor: Optional[int] = (
        int(cfg.data.get("prefetch_factor", 2)) if num_workers > 0 else None
    )
    persistent_workers = num_workers > 0

    train_ds = ImageFolderAlbu(
        root=str(root / "train"),
        transform=build_train_transform(cfg),
        subset_fraction=subset_fraction,
        subset_seed=subset_seed,
    )
    val_ds = ImageFolderAlbu(
        root=str(root / "val"),
        transform=build_val_transform(cfg),
        # Val set is always loaded in full regardless of subset_fraction
        # so accuracy metrics are not biased.
        subset_fraction=1.0,
        subset_seed=subset_seed,
    )

    common_kwargs: dict = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        drop_last=True,  # avoids a stale small batch corrupting grad accum
        **common_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size * 2,
        shuffle=False,
        drop_last=False,
        **common_kwargs,
    )

    return train_loader, val_loader
