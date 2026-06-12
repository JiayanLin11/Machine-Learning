"""Extract fold- and configuration-level metrics from experiment JSON files."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


OUTPUT_DIR = Path("outputs")
EXTRACTED_DIR = OUTPUT_DIR / "extracted"
EXCLUDED_DIRS = {"archive_old_coord", "archive_temp", "old_coord_backup", "extracted"}
CLASS_LABELS = tuple(range(5))
SCALAR_METRICS = ("qwk", "acc", "mae", "ece")
SUMMARY_METRICS = ("qwk", "acc", "mae", *(f"f1_KL{i}" for i in CLASS_LABELS))


def block_for(config_name: str) -> str:
    if config_name.startswith(("A0", "A1", "A2", "A3", "A4")):
        return "block2"
    if config_name.startswith("loss_"):
        return "block3"
    if config_name.startswith("attention_"):
        return "block4"
    if config_name.startswith("backbone_"):
        return "block5"
    if config_name.startswith(
        ("cdw_power_", "contrastive_lambda_", "contrastive_beta_", "crop_")
    ):
        return "block6"
    return "other"


def is_excluded(path: Path) -> bool:
    try:
        relative = path.relative_to(OUTPUT_DIR)
    except ValueError:
        return True
    return any(part in EXCLUDED_DIRS for part in relative.parts)


def details_paths() -> list[Path]:
    return sorted(
        path
        for path in OUTPUT_DIR.rglob("*_details.json")
        if not is_excluded(path)
    )


def config_name_for(details_path: Path) -> str:
    summary_path = details_path.with_name(
        details_path.name.replace("_details.json", "_summary.csv")
    )
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        if not summary.empty and "config_name" in summary.columns:
            return str(summary.iloc[0]["config_name"])

    stem = details_path.name.removesuffix("_details.json")
    parts = stem.split("_", 2)
    return parts[2] if len(parts) == 3 else stem


def explicit_f1_value(record: dict[str, Any], label: int) -> tuple[bool, float]:
    candidates = (
        f"f1_KL{label}",
        f"f1_class{label}",
        f"f1_class_{label}",
    )
    for key in candidates:
        if key in record:
            return True, float(record[key])
    return False, math.nan


def f1_values(record: dict[str, Any]) -> tuple[dict[int, float], list[int]]:
    values: dict[int, float] = {}
    explicit_found = False
    for label in CLASS_LABELS:
        found, value = explicit_f1_value(record, label)
        if found:
            explicit_found = True
            values[label] = value

    if explicit_found:
        missing = [label for label in CLASS_LABELS if label not in values]
        return values, missing

    raw = record.get("f1_per_class")
    if isinstance(raw, list):
        # metrics.py computes this array with labels=[0, 1, 2, 3, 4], so each
        # position has a fixed class identity rather than inferred alignment.
        for label in CLASS_LABELS:
            if label < len(raw) and raw[label] is not None:
                values[label] = float(raw[label])

    missing = [label for label in CLASS_LABELS if label not in values]
    return values, missing


def metric_row(
    block: str,
    config_name: str,
    record: dict[str, Any],
    prefix: str,
    include_fold: bool,
) -> tuple[dict[str, Any], list[int]]:
    row: dict[str, Any] = {
        "block": block,
        "config_name": config_name,
        "seed": record.get("seed"),
    }
    if include_fold:
        row["fold"] = record.get("fold")

    for metric in SCALAR_METRICS:
        row[f"{prefix}_{metric}"] = record.get(metric, math.nan)

    f1_by_label, missing = f1_values(record)
    for label in CLASS_LABELS:
        row[f"{prefix}_f1_KL{label}"] = f1_by_label.get(label, math.nan)
    return row, missing


def mean_std(series: pd.Series) -> str:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return "NaN"
    return f"{values.mean():.4f}±{values.std(ddof=0):.4f}"


def build_summary(val: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    keys = ["block", "config_name"]
    configs = pd.concat([val[keys], test[keys]], ignore_index=True).drop_duplicates()
    rows: list[dict[str, Any]] = []
    for _, config in configs.sort_values(keys).iterrows():
        block = config["block"]
        name = config["config_name"]
        val_group = val[(val["block"] == block) & (val["config_name"] == name)]
        test_group = test[(test["block"] == block) & (test["config_name"] == name)]
        row: dict[str, Any] = {
            "block": block,
            "config_name": name,
            "val_n_seed_fold": len(val_group),
            "test_n_seed": len(test_group),
        }
        for metric in SUMMARY_METRICS:
            row[f"val_{metric}_mean_std"] = mean_std(val_group[f"val_{metric}"])
            row[f"test_{metric}_mean_std"] = mean_std(test_group[f"test_{metric}"])
        rows.append(row)
    return pd.DataFrame(rows)


def print_results(
    val: pd.DataFrame,
    test: pd.DataFrame,
    missing: dict[tuple[str, str], set[str]],
) -> None:
    print("===== WARNINGS: missing per-class F1 =====")
    if not missing:
        print("None. Every included config has KL0-KL4 F1 in both val and test records.")
    else:
        for (block, config_name), messages in sorted(missing.items()):
            print(f"{block} -> {config_name}: " + "; ".join(sorted(messages)))

    print("\n===== METRICS BY BLOCK -> CONFIG =====")
    keys = ["block", "config_name"]
    configs = pd.concat([val[keys], test[keys]], ignore_index=True).drop_duplicates()
    for _, config in configs.sort_values(keys).iterrows():
        block = config["block"]
        name = config["config_name"]
        val_group = val[(val["block"] == block) & (val["config_name"] == name)]
        test_group = test[(test["block"] == block) & (test["config_name"] == name)]
        print(f"[{block}] {name}")
        print(
            "  val  "
            f"QWK {mean_std(val_group['val_qwk'])}  "
            f"ACC {mean_std(val_group['val_acc'])}  "
            f"MAE {mean_std(val_group['val_mae'])}  "
            + "  ".join(
                f"KL{i} {mean_std(val_group[f'val_f1_KL{i}'])}"
                for i in CLASS_LABELS
            )
        )
        print(
            "  test "
            f"QWK {mean_std(test_group['test_qwk'])}  "
            f"ACC {mean_std(test_group['test_acc'])}  "
            f"MAE {mean_std(test_group['test_mae'])}  "
            + "  ".join(
                f"KL{i} {mean_std(test_group[f'test_f1_KL{i}'])}"
                for i in CLASS_LABELS
            )
        )


def main() -> None:
    paths = details_paths()
    if not paths:
        raise SystemExit("No non-archived *_details.json files found under outputs/.")

    val_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    test_fold_rows: list[dict[str, Any]] = []
    missing: dict[tuple[str, str], set[str]] = {}

    for path in paths:
        payload = json.loads(path.read_text())
        result = payload.get("result", {})
        config_name = config_name_for(path)
        block = block_for(config_name)
        warning_key = (block, config_name)

        for record in result.get("val_raw", []):
            row, missing_labels = metric_row(
                block, config_name, record, "val", include_fold=True
            )
            val_rows.append(row)
            if missing_labels:
                missing.setdefault(warning_key, set()).add(
                    "val missing " + ",".join(f"KL{i}" for i in missing_labels)
                )

        for record in result.get("test_raw", []):
            has_fold = "fold" in record
            row, missing_labels = metric_row(
                block, config_name, record, "test", include_fold=has_fold
            )
            (test_fold_rows if has_fold else test_rows).append(row)
            if missing_labels:
                side = "test_per_fold" if has_fold else "test"
                missing.setdefault(warning_key, set()).add(
                    f"{side} missing "
                    + ",".join(f"KL{i}" for i in missing_labels)
                )

    val = pd.DataFrame(val_rows).sort_values(
        ["block", "config_name", "seed", "fold"], kind="stable"
    )
    test = pd.DataFrame(test_rows).sort_values(
        ["block", "config_name", "seed"], kind="stable"
    )
    summary = build_summary(val, test)

    targets: dict[Path, pd.DataFrame] = {
        EXTRACTED_DIR / "val_per_fold.csv": val,
        EXTRACTED_DIR / "test_per_seed.csv": test,
        EXTRACTED_DIR / "summary_by_config.csv": summary,
    }
    if test_fold_rows:
        test_per_fold = pd.DataFrame(test_fold_rows).sort_values(
            ["block", "config_name", "seed", "fold"], kind="stable"
        )
        targets[EXTRACTED_DIR / "test_per_fold.csv"] = test_per_fold

    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise SystemExit("Refusing to overwrite existing files: " + ", ".join(existing))

    print_results(val, test, missing)
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    for path, frame in targets.items():
        frame.to_csv(path, index=False)
    print("\nWritten:")
    for path, frame in targets.items():
        print(f"  {path} ({len(frame)} rows)")


if __name__ == "__main__":
    main()
