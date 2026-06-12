"""Experiment scheduler entry point."""

from __future__ import annotations

import argparse
import copy
import json
import tempfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

from knee_kl.calibration import TemperatureScaler, compare_ece
from knee_kl.config import Config
from knee_kl.datasets import (
    KneeKLDataset,
    build_loaders,
    build_transforms,
    load_all_samples,
    make_folds,
)
from knee_kl.metrics import compute_metrics, probabilities_from_logits
from knee_kl.train import evaluate, set_seed, train_one_fold
from knee_kl.tta import optimize_thresholds


SCALAR_METRICS = ("qwk", "acc", "mae", "macro_f1", "ece", "loss")
SOFTMAX_LOSSES = {"cdw_ce", "ce", "sord", "gaussian_soft"}


def set_global_seed(seed: int) -> None:
    """Set all training RNGs through the training module."""
    set_seed(seed)


def load_dev_and_test(cfg: Config) -> tuple[list[tuple[str, int, str]], list[tuple[str, int, str]]]:
    """Load train+val development samples and official held-out test samples."""
    return load_all_samples(cfg.data_root, "trainval"), load_all_samples(cfg.data_root, "test")


def _make_test_loader(test_samples: list[tuple[str, int, str]], cfg: Config) -> DataLoader:
    dataset = KneeKLDataset(test_samples, transform=build_transforms(cfg, train=False))
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, drop_last=False)


def _test_probabilities(model: torch.nn.Module, loader: DataLoader, device: torch.device, cfg: Config) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    metrics, logits, labels = evaluate(model, loader, device, return_logits=True, config=cfg)
    probs = probabilities_from_logits(logits.to(device), cfg).detach().cpu()
    return probs, logits, labels


def aggregate_metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate metric dictionaries.

    Scalar metrics are returned as {mean, std}. Per-class F1 and confusion are
    averaged element-wise to keep a table-friendly summary.
    """
    if not items:
        return {}
    out: dict[str, Any] = {}
    for key in SCALAR_METRICS:
        values = [float(item[key]) for item in items if key in item]
        if values:
            out[key] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=0))}
    if "f1_per_class" in items[0]:
        f1 = np.asarray([item["f1_per_class"] for item in items], dtype=float)
        out["f1_per_class"] = {"mean": f1.mean(axis=0).tolist(), "std": f1.std(axis=0, ddof=0).tolist()}
    if "confusion" in items[0]:
        confusion = np.asarray([item["confusion"] for item in items], dtype=float)
        out["confusion"] = confusion.mean(axis=0).tolist()
    return out


def _flatten_summary(config_name: str, cfg: Config, val_summary: dict[str, Any], test_summary: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "config_name": config_name,
        "backbone": cfg.backbone,
        "attention": cfg.attention,
        "loss_type": cfg.loss_type,
        "use_imbalance": cfg.use_imbalance,
        "use_contrastive": cfg.use_contrastive,
        "cdw_power": cfg.cdw_power,
        "contrastive_lambda": cfg.contrastive_lambda,
        "contrastive_beta": cfg.contrastive_beta,
        "center_crop_ratio": cfg.center_crop_ratio,
    }
    for prefix, summary in (("val", val_summary), ("test", test_summary)):
        for key in SCALAR_METRICS:
            if key in summary:
                row[f"{prefix}_{key}_mean"] = summary[key]["mean"]
                row[f"{prefix}_{key}_std"] = summary[key]["std"]
        if "f1_per_class" in summary:
            for idx, value in enumerate(summary["f1_per_class"]["mean"]):
                row[f"{prefix}_f1_class{idx}_mean"] = value
    return row


def _write_logs(config_name: str, cfg: Config, result: dict[str, Any]) -> tuple[Path, Path]:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{timestamp}_{config_name}"
    csv_path = output_dir / f"{stem}_summary.csv"
    json_path = output_dir / f"{stem}_details.json"
    row = _flatten_summary(config_name, cfg, result["val"], result["test"])
    pd.DataFrame([row]).to_csv(csv_path, index=False)
    serializable = json.loads(json.dumps(result, default=lambda value: value.tolist() if hasattr(value, "tolist") else str(value)))
    with json_path.open("w") as handle:
        json.dump({"config": asdict(cfg), "result": serializable}, handle, indent=2)
    return csv_path, json_path


def run_experiment(cfg: Config, seeds: Iterable[int] = (42, 1, 2), config_name: str = "experiment") -> dict[str, Any]:
    """Run multi-seed patient-aware CV and official-test fold ensembling."""
    dev_samples, test_samples = load_dev_and_test(cfg)
    if not dev_samples or not test_samples:
        raise ValueError("Both trainval development samples and official test samples are required.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = _make_test_loader(test_samples, cfg)
    print(f"test_loader_built_once samples={len(test_samples)} batch_size={cfg.batch_size}")

    val_metrics: list[dict[str, Any]] = []
    seed_test_metrics: list[dict[str, Any]] = []
    calibration_records: list[dict[str, Any]] = []
    seed_records: list[dict[str, Any]] = []

    for seed in seeds:
        cfg.seed = int(seed)
        set_global_seed(cfg.seed)
        folds = make_folds(dev_samples, cfg)
        fold_test_probs: list[torch.Tensor] = []
        test_labels_ref: torch.Tensor | None = None
        print(f"seed={cfg.seed} n_folds={len(folds)}")

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            print(f"seed={cfg.seed} fold={fold_idx} start")
            train_loader, val_loader = build_loaders(dev_samples, train_idx, val_idx, cfg)
            model = train_one_fold(
                cfg,
                train_loader,
                val_loader,
                device,
                config_name=config_name,
                fold=fold_idx,
            )

            val_eval, val_logits, val_labels = evaluate(model, val_loader, device, return_logits=True, config=cfg)
            val_eval["seed"] = cfg.seed
            val_eval["fold"] = fold_idx
            val_metrics.append(val_eval)

            test_probs, test_logits, test_labels = _test_probabilities(model, test_loader, device, cfg)
            fold_test_probs.append(test_probs)
            test_labels_ref = test_labels if test_labels_ref is None else test_labels_ref

            # Calibration is val-fit/test-eval. CORN is skipped because its logits are cumulative binary logits.
            if cfg.loss_type.lower() in SOFTMAX_LOSSES:
                scaler = TemperatureScaler()
                temperature = scaler.fit(val_logits.float(), val_labels.long())
                cal_summary, _bins = compare_ece(test_logits.float(), test_labels.long(), temperature)
                calibration_records.append({"seed": cfg.seed, "fold": fold_idx, **cal_summary})

        mean_test_probs = torch.stack(fold_test_probs, dim=0).mean(dim=0)
        assert test_labels_ref is not None
        test_pred = mean_test_probs.argmax(dim=1)
        test_eval = compute_metrics(test_labels_ref, test_pred, num_classes=cfg.num_classes)
        # ECE is computed from the seed-level 5-fold ensemble probabilities.
        from knee_kl.metrics import expected_calibration_error
        test_eval["ece"] = expected_calibration_error(mean_test_probs, test_labels_ref)
        test_eval["seed"] = cfg.seed
        seed_test_metrics.append(test_eval)
        seed_records.append({"seed": cfg.seed, "folds_ensembled": len(fold_test_probs)})
        print(f"seed={cfg.seed} test_ensemble_folds={len(fold_test_probs)} qwk={test_eval['qwk']:.4f}")

    result = {
        "val": aggregate_metrics(val_metrics),
        "test": aggregate_metrics(seed_test_metrics),
        "val_raw": val_metrics,
        "test_raw": seed_test_metrics,
        "calibration": calibration_records,
        "seed_records": seed_records,
    }
    csv_path, json_path = _write_logs(config_name, cfg, result)
    result["log_csv"] = str(csv_path)
    result["log_json"] = str(json_path)
    print(f"logs_written csv={csv_path} json={json_path}")
    print(f"val_summary={result['val']}")
    print(f"test_summary={result['test']}")
    return result


def run_threshold_analysis(model: torch.nn.Module, val_loader: DataLoader, device: torch.device, cfg: Config) -> dict[str, Any]:
    """Optional add-on analysis for final A4, not part of the main protocol."""
    metrics, logits, labels = evaluate(model, val_loader, device, return_logits=True, config=cfg)
    probs = probabilities_from_logits(logits.to(device), cfg).detach().cpu()
    result = optimize_thresholds(probs, labels, num_classes=cfg.num_classes)
    print(f"threshold_analysis={result}")
    return result


def _clone_config(base: Config, **updates: Any) -> Config:
    values = asdict(base)
    values.update(updates)
    return Config(**values)


def build_block_configs(block: str, base: Config) -> list[tuple[str, Config, tuple[int, ...]]]:
    """Build experiment blocks. Default CLI block is ablation (block 2)."""
    configs: list[tuple[str, Config, tuple[int, ...]]] = []
    if block in {"ablation", "all"}:
        # Block 2: cumulative core ablation A0..A4, full 3 seeds for paper-critical numbers.
        configs.extend([
            ("A0_ce_plain", _clone_config(base, attention="none", loss_type="ce", use_imbalance=False, use_contrastive=False), (42, 1, 2)),
            ("A1_cdw", _clone_config(base, attention="none", loss_type="cdw_ce", use_imbalance=False, use_contrastive=False), (42, 1, 2)),
            ("A2_triplet", _clone_config(base, attention="triplet", loss_type="cdw_ce", use_imbalance=False, use_contrastive=False), (42, 1, 2)),
            ("A3_imbalance", _clone_config(base, attention="triplet", loss_type="cdw_ce", use_imbalance=True, use_contrastive=False), (42, 1, 2)),
            ("A4_full", _clone_config(base), (42, 1, 2)),
        ])
    if block in {"loss", "all"}:
        # Block 3: loss comparison, fixed triplet and no imbalance/contrastive; single seed for scouting.
        for loss_type in ("ce", "mse", "corn", "sord", "gaussian_soft", "cdw_ce"):
            configs.append((f"loss_{loss_type}", _clone_config(base, attention="triplet", loss_type=loss_type, use_imbalance=False, use_contrastive=False), (42,)))
    if block in {"attention", "all"}:
        # Block 4: attention comparison, fixed cdw_ce and no imbalance/contrastive; single seed for scouting.
        for attention in ("none", "se", "cbam", "triplet", "coord"):
            configs.append((f"attention_{attention}", _clone_config(base, attention=attention, loss_type="cdw_ce", use_imbalance=False, use_contrastive=False), (42,)))
    if block in {"backbone", "all"}:
        # Block 5: backbone comparison under full A4 recipe; single seed for scouting.
        for backbone in ("resnet50", "densenet121", "efficientnet_b0", "swin_tiny_patch4_window7_224", "convnext_tiny"):
            configs.append((f"backbone_{backbone}", _clone_config(base, backbone=backbone), (42,)))
    if block in {"sensitivity", "all"}:
        # Block 6: A4-prime hyperparameter sensitivity, single seed to inspect trends.
        sensitivity_base = _clone_config(
            base,
            attention="triplet",
            loss_type="cdw_ce",
            use_imbalance=True,
            use_oversampling=False,
            use_class_weight=True,
            use_contrastive=True,
        )
        for value in (2, 3, 4, 5, 6):
            configs.append((f"cdw_power_{value}", _clone_config(sensitivity_base, cdw_power=float(value)), (42,)))
        for value in (0.1, 0.3, 0.5, 1.0):
            configs.append((f"contrastive_lambda_{value}", _clone_config(sensitivity_base, contrastive_lambda=float(value)), (42,)))
        for value in (0.5, 1.0, 2.0):
            configs.append((f"contrastive_beta_{value}", _clone_config(sensitivity_base, contrastive_beta=float(value)), (42,)))
        for config_name, lr_backbone, lr_head in (
            ("lr_0.3x", 3e-5, 3e-4),
            ("lr_1x", 1e-4, 1e-3),
            ("lr_3x", 3e-4, 3e-3),
        ):
            configs.append((
                config_name,
                _clone_config(sensitivity_base, lr_backbone=lr_backbone, lr_head=lr_head),
                (42,),
            ))
    return configs


def _write_png(path: Path, seed: int, size: int = 72) -> None:
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, size=(size, size), dtype=np.uint8), mode="L").save(path)


def _make_smoke_dataset(root: Path) -> None:
    for split in ("train", "val", "test", "auto_test"):
        for label in range(5):
            (root / split / str(label)).mkdir(parents=True, exist_ok=True)
    for label in range(5):
        for idx in range(4):
            patient = 9100000 + label * 10 + idx
            split = "train" if idx < 3 else "val"
            side = "L" if idx % 2 == 0 else "R"
            _write_png(root / split / str(label) / f"{patient:07d}{side}.png", patient)
        for idx in range(2):
            patient = 9200000 + label * 10 + idx
            side = "L" if idx == 0 else "R"
            _write_png(root / "test" / str(label) / f"{patient:07d}{side}.png", patient)


def run_smoke() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _make_smoke_dataset(root)
        cfg = Config(
            data_root=str(root),
            output_dir="outputs",
            img_size=64,
            center_crop_ratio=1.0,
            batch_size=4,
            epochs=1,
            early_stop_patience=1,
            num_workers=0,
            n_folds=2,
            attention="none",
            loss_type="ce",
            use_imbalance=False,
            use_contrastive=False,
        )
        return run_experiment(cfg, seeds=(0, 1), config_name="smoke_A0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Knee KL grading experiment scheduler")
    parser.add_argument("--block", default="ablation", choices=["ablation", "loss", "attention", "backbone", "sensitivity", "all"])
    parser.add_argument("--smoke", action="store_true", help="Run a tiny synthetic scheduling smoke test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        result = run_smoke()
        print(f"smoke_csv={result['log_csv']}")
        print(Path(result["log_csv"]).read_text())
        return

    base = Config(epochs=50, early_stop_patience=10)
    # Default is block 2 only. Other blocks are selectable with --block and use explicit seed budgets.
    for config_name, cfg, seeds in build_block_configs(args.block, base):
        print(f"running config={config_name} seeds={seeds}")
        run_experiment(cfg, seeds=seeds, config_name=config_name)


if __name__ == "__main__":
    main()
