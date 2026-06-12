"""Aggregate fold-level experiment details into per-seed metrics."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


OUTPUT_DIR = Path("outputs")
TARGET_PATH = OUTPUT_DIR / "extracted" / "per_config_per_seed.csv"
CLASS_LABELS = range(5)
BATCH_SUFFIX_RE = re.compile(r"(?:_seeds\d+|_\d+seeds?)$", re.IGNORECASE)


def is_archived(path: Path) -> bool:
    relative = path.relative_to(OUTPUT_DIR)
    directory_parts = relative.parts[:-1]
    return any(
        part == "old_coord_backup" or "archive" in part.lower()
        for part in directory_parts
    )


def details_paths() -> list[Path]:
    return sorted(
        path
        for path in OUTPUT_DIR.rglob("*_details.json")
        if not is_archived(path)
    )


def raw_config_name(path: Path) -> str:
    summary_path = path.with_name(path.name.replace("_details.json", "_summary.csv"))
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        if not summary.empty and "config_name" in summary.columns:
            return str(summary.iloc[0]["config_name"])

    stem = path.name.removesuffix("_details.json")
    parts = stem.split("_", 2)
    return parts[2] if len(parts) == 3 else stem


def normalized_config_name(name: str) -> str:
    previous = None
    normalized = name
    while normalized != previous:
        previous = normalized
        normalized = BATCH_SUFFIX_RE.sub("", normalized)
    return normalized


def f1_values(record: dict[str, Any]) -> list[float]:
    raw = record.get("f1_per_class")
    if isinstance(raw, list):
        return [
            float(raw[label]) if label < len(raw) and raw[label] is not None else math.nan
            for label in CLASS_LABELS
        ]

    values = []
    for label in CLASS_LABELS:
        value = math.nan
        for key in (f"f1_KL{label}", f"f1_class{label}", f"f1_class_{label}"):
            if key in record:
                value = float(record[key])
                break
        values.append(value)
    return values


def mean_metric(records: list[dict[str, Any]], key: str) -> float:
    values = pd.to_numeric(
        pd.Series([record.get(key, math.nan) for record in records]),
        errors="coerce",
    )
    return float(values.mean()) if values.notna().any() else math.nan


def build_row(
    config_name: str,
    seed: int,
    val_records: list[dict[str, Any]],
    test_record: dict[str, Any] | None,
) -> dict[str, Any]:
    val_records = sorted(val_records, key=lambda record: record.get("fold", -1))
    val_qwk_values = pd.to_numeric(
        pd.Series([record.get("qwk", math.nan) for record in val_records]),
        errors="coerce",
    )
    row: dict[str, Any] = {
        "config_name": config_name,
        "seed": seed,
        "n_folds": len(val_records),
        "n_collapsed_folds": int((val_qwk_values < 1e-6).sum()),
        "val_qwk": mean_metric(val_records, "qwk"),
    }

    for label in CLASS_LABELS:
        values = [f1_values(record)[label] for record in val_records]
        numeric = pd.to_numeric(pd.Series(values), errors="coerce")
        row[f"val_f1_KL{label}"] = (
            float(numeric.mean()) if numeric.notna().any() else math.nan
        )

    test_record = test_record or {}
    for metric in ("qwk", "acc", "mae", "ece"):
        row[f"test_{metric}"] = test_record.get(metric, math.nan)
    test_f1 = f1_values(test_record)
    for label in CLASS_LABELS:
        row[f"test_f1_KL{label}"] = test_f1[label]
    return row


def print_mapping(mapping: dict[str, str]) -> None:
    print("===== CONFIG NAME MAPPING =====")
    for raw_name, display_name in sorted(mapping.items()):
        marker = "" if raw_name == display_name else "  [normalized]"
        print(f"{raw_name} -> {display_name}{marker}")


def print_compact(frame: pd.DataFrame) -> None:
    print("\n===== PER CONFIG PER SEED (COMPACT) =====")
    compact = frame[
        [
            "config_name",
            "seed",
            "n_collapsed_folds",
            "val_qwk",
            "test_qwk",
            "test_f1_KL0",
            "test_f1_KL1",
        ]
    ].copy()
    print(
        compact.to_string(
            index=False,
            na_rep="NaN",
            formatters={
                "val_qwk": lambda value: f"{value:.4f}",
                "test_qwk": lambda value: f"{value:.4f}",
                "test_f1_KL0": lambda value: f"{value:.4f}",
                "test_f1_KL1": lambda value: f"{value:.4f}",
            },
        )
    )


def main() -> None:
    paths = details_paths()
    if not paths:
        raise SystemExit("No non-archived *_details.json files found under outputs/.")

    mapping: dict[str, str] = {}
    runs: dict[tuple[str, int], dict[str, Any]] = {}
    duplicates: list[tuple[str, int, Path, Path]] = []

    for path in paths:
        payload = json.loads(path.read_text())
        result = payload.get("result", {})
        raw_name = raw_config_name(path)
        display_name = normalized_config_name(raw_name)
        mapping[raw_name] = display_name

        val_by_seed: dict[int, list[dict[str, Any]]] = {}
        for record in result.get("val_raw", []):
            if record.get("seed") is not None:
                val_by_seed.setdefault(int(record["seed"]), []).append(record)

        test_by_seed = {
            int(record["seed"]): record
            for record in result.get("test_raw", [])
            if record.get("seed") is not None and "fold" not in record
        }

        for seed in sorted(set(val_by_seed) | set(test_by_seed)):
            key = (display_name, seed)
            candidate = {
                "path": path,
                "val": val_by_seed.get(seed, []),
                "test": test_by_seed.get(seed),
            }
            if key in runs:
                duplicates.append((display_name, seed, path, runs[key]["path"]))
            runs[key] = candidate

    print_mapping(mapping)
    print("\n===== DUPLICATE CONFIG/SEED RUNS =====")
    if duplicates:
        for config_name, seed, kept, dropped in duplicates:
            print(
                f"{config_name}, seed={seed}: kept {kept.name}; "
                f"dropped {dropped.name}"
            )
    else:
        print("None.")

    rows = [
        build_row(config_name, seed, run["val"], run["test"])
        for (config_name, seed), run in runs.items()
    ]
    frame = pd.DataFrame(rows).sort_values(
        ["config_name", "seed"],
        kind="stable",
    )

    TARGET_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TARGET_PATH, index=False)
    print_compact(frame)
    print(f"\nWritten: {TARGET_PATH} ({len(frame)} rows)")


if __name__ == "__main__":
    main()
