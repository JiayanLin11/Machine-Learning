"""Data loading utilities for knee KL grading."""

from __future__ import annotations

import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import transforms


FILENAME_RE = re.compile(r"^(?P<patient_id>\d{7})(?P<side>[LR])\.png$", re.IGNORECASE)
VALID_SPLITS = {"train", "val", "test", "trainval"}


def parse_patient_id(filename: str) -> str:
    """Parse the 7-digit patient ID from ``[patient_id][L/R].png``."""
    match = FILENAME_RE.match(os.path.basename(filename))
    if match is None:
        raise ValueError(f"Invalid knee image filename: {filename!r}")
    return match.group("patient_id")


def _parse_filename(filename: str) -> tuple[str, str]:
    match = FILENAME_RE.match(os.path.basename(filename))
    if match is None:
        raise ValueError(f"Invalid knee image filename: {filename!r}")
    return match.group("patient_id"), match.group("side").upper()


def _split_dirs(data_root: str, split: str) -> list[Path]:
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {sorted(VALID_SPLITS)}, got {split!r}")

    root = Path(os.path.expanduser(data_root))
    names = ["train", "val"] if split == "trainval" else [split]
    return [root / name for name in names]


def load_all_samples(data_root: str, split: str) -> list[tuple[str, int, str]]:
    """Scan samples for a split.

    Args:
        data_root: Dataset root containing ``train/``, ``val/``, ``test/`` and
            ``auto_test/`` directories. ``~`` is expanded.
        split: One of ``trainval``, ``train``, ``val`` or ``test``. ``trainval``
            merges official train and val directories. ``auto_test`` is ignored.

    Returns:
        A sorted list of ``(img_path, kl_label, patient_id)`` tuples.
    """
    samples: list[tuple[str, int, str]] = []

    for split_dir in _split_dirs(data_root, split):
        if split_dir.name == "auto_test":
            continue
        if not split_dir.exists():
            continue

        for label_dir in sorted(split_dir.iterdir(), key=lambda p: p.name):
            if not label_dir.is_dir() or label_dir.name not in {"0", "1", "2", "3", "4"}:
                continue

            label = int(label_dir.name)
            for img_path in sorted(label_dir.glob("*.png")):
                patient_id, _side = _parse_filename(img_path.name)
                samples.append((str(img_path), label, patient_id))

    return samples


def build_dataframe(data_root: str) -> list[tuple[str, int, str]]:
    """Compatibility wrapper returning the train+val development sample list."""
    return load_all_samples(data_root, "trainval")


class CenterCropRatio:
    """Crop the centered image region by ratio before downstream resizing."""

    def __init__(self, ratio: float) -> None:
        if ratio <= 0 or ratio > 1:
            raise ValueError(f"center_crop_ratio must be in (0, 1], got {ratio}")
        self.ratio = ratio

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.ratio == 1:
            return image

        width, height = image.size
        crop_width = max(1, int(round(width * self.ratio)))
        crop_height = max(1, int(round(height * self.ratio)))
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        return image.crop((left, top, left + crop_width, top + crop_height))


def build_transforms(config: Any, train: bool | str) -> transforms.Compose:
    """Build image transforms for training or evaluation."""
    if isinstance(train, str):
        train = train.lower() == "train"

    ops: list[Any] = []
    center_crop_ratio = float(getattr(config, "center_crop_ratio", 1.0))
    if center_crop_ratio < 1.0:
        ops.append(CenterCropRatio(center_crop_ratio))

    ops.append(transforms.Resize((config.img_size, config.img_size)))

    if train:
        ops.extend(
            [
                transforms.RandomRotation(degrees=10),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
            ]
        )

    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    return transforms.Compose(ops)


class KneeKLDataset(Dataset):
    """Dataset returning normalized image tensors and integer KL labels."""

    def __init__(self, samples: Sequence[tuple[str, int, str]], transform: Any | None = None) -> None:
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        img_path, label, _patient_id = self.samples[index]
        with Image.open(img_path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, int(label)


def _sample_fields(samples: Sequence[tuple[str, int, str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray([sample[1] for sample in samples])
    groups = np.asarray([sample[2] for sample in samples])
    indices = np.arange(len(samples))
    return indices, labels, groups


def make_folds(
    samples: Sequence[tuple[str, int, str]],
    config_or_n_folds: Any,
    patient_aware_split: bool | None = None,
    seed: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create stratified folds.

    When ``patient_aware_split`` is false this falls back to ``StratifiedKFold``;
    that mode can leak patient identity across train and validation folds.
    """
    if hasattr(config_or_n_folds, "n_folds"):
        n_folds = int(config_or_n_folds.n_folds)
        patient_aware = bool(getattr(config_or_n_folds, "patient_aware_split", True))
        random_seed = int(getattr(config_or_n_folds, "seed", 42))
    else:
        n_folds = int(config_or_n_folds)
        patient_aware = True if patient_aware_split is None else bool(patient_aware_split)
        random_seed = 42 if seed is None else int(seed)

    indices, labels, groups = _sample_fields(samples)

    if patient_aware:
        splitter = StratifiedGroupKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=random_seed,
        )
        return [(train_idx, val_idx) for train_idx, val_idx in splitter.split(indices, labels, groups)]

    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    return [(train_idx, val_idx) for train_idx, val_idx in splitter.split(indices, labels)]


def build_sampler(train_labels: Sequence[int], config: Any) -> WeightedRandomSampler | None:
    """Build inverse-frequency sampler for imbalanced KL classes."""
    if not (
        getattr(config, "use_imbalance", False)
        and getattr(config, "use_oversampling", True)
    ):
        return None

    labels = [int(label) for label in train_labels]
    counts = Counter(labels)
    weights = torch.as_tensor([1.0 / counts[label] for label in labels], dtype=torch.double)
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def build_loaders(
    samples: Sequence[tuple[str, int, str]],
    train_idx: Iterable[int],
    val_idx: Iterable[int],
    config: Any,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders for one fold."""
    train_idx = list(train_idx)
    val_idx = list(val_idx)
    train_labels = [samples[index][1] for index in train_idx]

    train_dataset = KneeKLDataset(samples, transform=build_transforms(config, train=True))
    val_dataset = KneeKLDataset(samples, transform=build_transforms(config, train=False))
    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(val_dataset, val_idx)

    sampler = build_sampler(train_labels, config)
    train_loader = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=config.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    return train_loader, val_loader


def create_dataset(dataframe: Sequence[tuple[str, int, str]], config: Any, split: str) -> KneeKLDataset:
    """Compatibility wrapper for constructing a dataset from sample tuples."""
    return KneeKLDataset(dataframe, transform=build_transforms(config, split))


def create_dataloader(dataset: Dataset, config: Any, shuffle: bool) -> DataLoader:
    """Compatibility wrapper for a plain DataLoader."""
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
    )


def _write_synthetic_image(path: Path, size: int = 96) -> None:
    array = np.random.default_rng(abs(hash(str(path))) % (2**32)).integers(
        0,
        256,
        size=(size, size),
        dtype=np.uint8,
    )
    Image.fromarray(array, mode="L").save(path)


def _make_synthetic_dataset(root: Path) -> None:
    patient_ids = [f"{9000000 + idx:07d}" for idx in range(30)]

    for split in ("train", "val", "auto_test"):
        for label in range(5):
            (root / split / str(label)).mkdir(parents=True, exist_ok=True)

    for label in range(5):
        for offset in range(6):
            patient_id = patient_ids[label * 6 + offset]
            split = "train" if offset < 4 else "val"
            side = "L" if offset % 2 == 0 else "R"
            _write_synthetic_image(root / split / str(label) / f"{patient_id}{side}.png")

    shared_cases = [
        ("9000000", "R", 1, "val"),
        ("9000001", "L", 2, "train"),
        ("9000006", "R", 3, "val"),
        ("9000012", "L", 4, "train"),
        ("9000018", "R", 0, "val"),
    ]
    for patient_id, side, label, split in shared_cases:
        _write_synthetic_image(root / split / str(label) / f"{patient_id}{side}.png")

    _write_synthetic_image(root / "auto_test" / "4" / "9999999R.png")


def _distribution(samples: Sequence[tuple[str, int, str]], indices: Iterable[int]) -> dict[int, int]:
    counts = Counter(samples[index][1] for index in indices)
    return {label: counts.get(label, 0) for label in range(5)}


if __name__ == "__main__":
    from knee_kl.config import Config

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _make_synthetic_dataset(root)

        cfg = Config(data_root=str(root), batch_size=8, num_workers=0, use_imbalance=True)
        samples = load_all_samples(cfg.data_root, "trainval")
        folds = make_folds(samples, cfg)

        print(f"samples: {len(samples)}")
        try:
            load_all_samples(cfg.data_root, "auto_test")
        except ValueError as error:
            print(f"auto_test rejected: True ({error})")
        else:
            print("auto_test rejected: False")

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            train_patients = {samples[index][2] for index in train_idx}
            val_patients = {samples[index][2] for index in val_idx}
            no_leakage = train_patients.isdisjoint(val_patients)
            val_counts = _distribution(samples, val_idx)
            missing_val_classes = [label for label, count in val_counts.items() if count == 0]
            print(
                f"fold {fold_idx}: "
                f"train={len(train_idx)} val={len(val_idx)} "
                f"train_dist={_distribution(samples, train_idx)} "
                f"val_class_counts={val_counts} "
                f"no_patient_leakage={no_leakage}"
            )
            if missing_val_classes:
                print(
                    f"warning: fold {fold_idx} validation has zero samples for "
                    f"classes {missing_val_classes}"
                )

        train_loader, val_loader = build_loaders(samples, folds[0][0], folds[0][1], cfg)
        print(f"train_loader.drop_last: {train_loader.drop_last}")
        print(f"val_loader.drop_last: {val_loader.drop_last}")
        print(f"train_sampler: {type(train_loader.sampler).__name__}")
        images, labels = next(iter(train_loader))
        print(f"batch images shape: {list(images.shape)}")
        print(f"batch labels shape: {list(labels.shape)}")
