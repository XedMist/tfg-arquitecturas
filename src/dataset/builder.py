from __future__ import annotations

import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional

import torch
import torchvision.datasets as tvdatasets
import torchvision.transforms as T
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


class ImageFolderAlbu(Dataset):
    """ImageFolder-compatible dataset backed by torchvision transforms.

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
        transform:        ``torchvision.transforms.Compose`` pipeline.
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

        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class ImagenetteDataset(Dataset):
    """TorchImagenette with torchvision transforms."""

    _URL = "https://s3.amazonaws.com/fast-ai-modelzoo/imagenette2.tgz"
    _VALID_EXT: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp"})

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform: Optional[Callable] = None,
        subset_fraction: float = 1.0,
        subset_seed: int = 42,
    ) -> None:
        if not (0.0 < subset_fraction <= 1.0):
            raise ValueError(
                f"subset_fraction must be in (0, 1], got {subset_fraction}"
            )

        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.subset_fraction = subset_fraction
        self.subset_seed = subset_seed
        self.classes: list[str] = []

        imagenette2_path = Path(root) / "imagenette2"
        download = not imagenette2_path.exists()

        self._dataset = tvdatasets.Imagenette(
            root=str(root),
            split=split,
            download=download,
        )
        self._load_samples()

    def _load_samples(self) -> None:
        class_folders = sorted(
            d for d in (self.root / self.split).iterdir() if d.is_dir()
        )
        if not class_folders:
            raise RuntimeError(
                f"No class folders found in '{self.root / self.split}'. "
                f"Ensure Imagenette is downloaded."
            )

        self.classes = [d.name for d in class_folders]
        class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        samples_per_class: dict[int, list[tuple[str, int]]] = defaultdict(list)
        for cls_dir in class_folders:
            idx = class_to_idx[cls_dir.name]
            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() in self._VALID_EXT:
                    samples_per_class[idx].append((str(img_path), idx))

        if not any(samples_per_class.values()):
            raise RuntimeError(f"No valid images in '{self.root / self.split}'.")

        rng = random.Random(self.subset_seed)
        all_samples: list[tuple[str, int]] = []

        for cls_idx in sorted(samples_per_class.keys()):
            cls_samples = sorted(samples_per_class[cls_idx], key=lambda x: x[0])
            n_keep = max(1, math.ceil(len(cls_samples) * self.subset_fraction))
            all_samples.extend(rng.sample(cls_samples, n_keep))

        self.samples = all_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]

        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# ---------------------------------------------------------------------------
# Transform factories
# ---------------------------------------------------------------------------


def build_train_transform(cfg: DictConfig) -> T.Compose:
    """DeiT / Swin training augmentation pipeline using torchvision."""
    aug = cfg.augmentation.train
    data = cfg.data

    crop_size: int = int(aug.random_resized_crop)
    brightness: float = float(aug.color_jitter.brightness)
    contrast: float = float(aug.color_jitter.contrast)
    saturation: float = float(aug.color_jitter.saturation)
    hue: float = float(aug.color_jitter.hue)
    color_jitter_p: float = float(aug.color_jitter.p)

    # torchvision ColorJitter uses [0, 1] ranges differently than albumentations
    # albumentations uses ±limit symmetrically, torchvision uses absolute jitter values
    # Convert: albumentations brightness=0.4 means ±0.4 range → torchvision brightness=0.4
    jitter = T.ColorJitter(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
    )

    transforms: list[Callable] = [
        T.RandomResizedCrop(
            crop_size,
            scale=(0.08, 1.0),
            ratio=(3.0 / 4.0, 4.0 / 3.0),
            interpolation=T.InterpolationMode.BILINEAR,
        ),
        T.RandomHorizontalFlip(p=float(aug.horizontal_flip_p)),
        T.RandomApply([jitter], p=color_jitter_p),
    ]

    # Optional Gaussian blur (DeiT-III adds this at p=0.1).
    gaussian_blur_p: float = float(aug.get("gaussian_blur_p", 0.0))
    if gaussian_blur_p > 0.0:
        transforms.append(
            T.RandomApply(
                [T.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))],
                p=gaussian_blur_p,
            )
        )

    # Random grayscale (DeiT uses p=0.2).
    grayscale_p: float = float(aug.get("grayscale_p", 0.0))
    if grayscale_p > 0.0:
        transforms.append(T.RandomGrayscale(p=grayscale_p))

    # ── RandAugment / AutoAugment (PIL-level, before ToTensor) ────────────────
    # Policy string e.g. "rand-m9-mstd0.5-inc1" (timm format) or
    # "randaugment" / "trivialaugment" (torchvision fallback).
    auto_augment_policy: str = str(aug.get("auto_augment", "") or "")
    if auto_augment_policy:
        aa_transform = _build_auto_augment(auto_augment_policy, crop_size)
        if aa_transform is not None:
            transforms.append(aa_transform)

    transforms.extend(
        [
            T.ToTensor(),
            T.Normalize(mean=list(data.mean), std=list(data.std)),
        ]
    )

    # ── Random Erasing (tensor-level, after Normalize) ────────────────────
    re_prob: float = float(aug.get("re_prob", 0.0))
    if re_prob > 0.0:
        transforms.append(
            T.RandomErasing(
                p=re_prob,
                scale=(0.02, 0.33),
                ratio=(0.3, 3.3),
                value="random",  # random noise patch (timm default)
            )
        )

    return T.Compose(transforms)


def _build_auto_augment(policy: str, img_size: int):
    """Build a RandAugment / AutoAugment transform from a policy string.

    Tries timm first (supports the full "rand-m9-mstd0.5-inc1" syntax).
    Falls back to torchvision equivalents for simple policy names.
    Returns None if the policy string is unrecognised.
    """
    policy_lower = policy.lower()
    try:
        from timm.data.auto_augment import rand_augment_transform  # type: ignore

        aa_params = {"translate_const": int(img_size * 0.45), "img_mean": (128, 128, 128)}
        return rand_augment_transform(policy, aa_params)
    except (ImportError, Exception):
        pass

    # timm not available or policy unrecognised — fall back to torchvision
    if "trivialaugment" in policy_lower:
        return T.TrivialAugmentWide()
    if "randaugment" in policy_lower or policy_lower.startswith("rand"):
        return T.RandAugment(num_ops=2, magnitude=9)
    if "autoaugment" in policy_lower:
        return T.AutoAugment(T.AutoAugmentPolicy.IMAGENET)

    import logging
    logging.getLogger(__name__).warning(
        "auto_augment policy '%s' unrecognised and timm unavailable — skipping.", policy
    )
    return None


def build_val_transform(cfg: DictConfig) -> T.Compose:
    """Standard ImageNet validation pipeline using torchvision.

    Resize the shortest side to ``resize`` (typically 256 for 224-crop
    models), then centre-crop to ``center_crop`` (typically 224).
    This matches the torchvision / timm evaluation convention.
    """
    aug = cfg.augmentation.val
    data = cfg.data

    resize_size: int = int(aug.resize)
    center_crop_size: int = int(aug.center_crop)

    return T.Compose(
        [
            T.Resize(resize_size, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(center_crop_size),
            T.ToTensor(),
            T.Normalize(mean=list(data.mean), std=list(data.std)),
        ]
    )


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------


def build_imagenette_loaders(
    cfg: DictConfig,
) -> tuple[DataLoader, DataLoader]:
    """Build train/val loaders from torchvision Imagenette."""
    root = Path(cfg.data.root)
    subset_fraction = float(cfg.data.get("subset_fraction", 1.0))
    subset_seed = int(cfg.data.get("subset_seed", 42))
    num_workers = int(cfg.data.workers)
    pin_memory = bool(cfg.data.pin_memory)
    prefetch_factor: Optional[int] = (
        int(cfg.data.get("prefetch_factor", 2)) if num_workers > 0 else None
    )
    persistent_workers = num_workers > 0

    imagenette2_path = Path(root) / "imagenette2"
    download = not imagenette2_path.exists()

    train_ds = tvdatasets.Imagenette(
        root=str(root),
        split="train",
        transform=build_train_transform(cfg),
        download=download,
    )
    val_ds = tvdatasets.Imagenette(
        root=str(root),
        split="val",
        transform=build_val_transform(cfg),
        download=False,
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
        drop_last=True,
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


def build_classification_loaders(
    cfg: DictConfig,
) -> tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders.

    Dispatches to imagenette loader if cfg.data.dataset_type == "imagenette",
    otherwise uses ImageFolderAlbu from local directory.
    """
    dataset_type = cfg.data.get("dataset_type", "imagefolder")

    if dataset_type == "imagenette":
        return build_imagenette_loaders(cfg)

    # Default: imagefolder (original behavior)
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
