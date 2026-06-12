"""Evaluation metrics for KL grading."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from coral_pytorch.dataset import corn_label_from_logits
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
)


def _as_numpy(values: Any) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values)


def probabilities_from_logits(logits: torch.Tensor, config: Any) -> torch.Tensor:
    """Convert model logits to class probabilities for softmax losses and CORN."""
    if config.loss_type.lower() != "corn":
        return F.softmax(logits, dim=1)

    # CORN class probabilities are recovered from cumulative continuation probabilities.
    continuation = torch.cumprod(torch.sigmoid(logits), dim=1)
    probs = torch.zeros(logits.shape[0], config.num_classes, device=logits.device, dtype=logits.dtype)
    probs[:, 0] = 1.0 - continuation[:, 0]
    if config.num_classes > 2:
        probs[:, 1:-1] = continuation[:, :-1] - continuation[:, 1:]
    probs[:, -1] = continuation[:, -1]
    return probs.clamp_min(0.0)


def compute_accuracy(y_true: Any, y_pred: Any) -> float:
    """Compute classification accuracy."""
    return float(accuracy_score(_as_numpy(y_true), _as_numpy(y_pred)))


def compute_mae(y_true: Any, y_pred: Any) -> float:
    """Compute mean absolute error over ordinal class indices."""
    return float(mean_absolute_error(_as_numpy(y_true), _as_numpy(y_pred)))


def compute_qwk(y_true: Any, y_pred: Any, num_classes: int = 5) -> float:
    """Compute quadratic weighted Cohen kappa with fixed class labels."""
    labels = list(range(num_classes))
    score = cohen_kappa_score(_as_numpy(y_true), _as_numpy(y_pred), labels=labels, weights="quadratic")
    return float(0.0 if np.isnan(score) else score)


def compute_confusion_matrix(y_true: Any, y_pred: Any, num_classes: int = 5) -> np.ndarray:
    """Compute a fixed-size confusion matrix."""
    return confusion_matrix(_as_numpy(y_true), _as_numpy(y_pred), labels=list(range(num_classes)))


def compute_metrics(y_true: Any, y_pred: Any, num_classes: int = 5) -> dict[str, Any]:
    """Compute the project metric bundle with fixed class positions."""
    y_true_np = _as_numpy(y_true).astype(int)
    y_pred_np = _as_numpy(y_pred).astype(int)
    labels = list(range(num_classes))
    f1_per_class = f1_score(y_true_np, y_pred_np, labels=labels, average=None, zero_division=0)
    macro_f1 = f1_score(y_true_np, y_pred_np, labels=labels, average="macro", zero_division=0)
    return {
        "qwk": compute_qwk(y_true_np, y_pred_np, num_classes=num_classes),
        "acc": compute_accuracy(y_true_np, y_pred_np),
        "mae": compute_mae(y_true_np, y_pred_np),
        "f1_per_class": f1_per_class.tolist(),
        "macro_f1": float(macro_f1),
        "confusion": compute_confusion_matrix(y_true_np, y_pred_np, num_classes=num_classes),
    }


def expected_calibration_error(probs: Any, y_true: Any, n_bins: int = 15) -> float:
    """Compute confidence ECE using predicted-class probabilities."""
    probs_np = _as_numpy(probs).astype(float)
    y_true_np = _as_numpy(y_true).astype(int)
    if probs_np.size == 0:
        return 0.0
    confidences = probs_np.max(axis=1)
    predictions = probs_np.argmax(axis=1)
    correctness = (predictions == y_true_np).astype(float)
    ece = 0.0
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    for bin_idx in range(n_bins):
        lower = bin_edges[bin_idx]
        upper = bin_edges[bin_idx + 1]
        in_bin = (confidences >= lower) & (confidences <= upper if bin_idx == n_bins - 1 else confidences < upper)
        if not np.any(in_bin):
            continue
        ece += in_bin.mean() * abs(correctness[in_bin].mean() - confidences[in_bin].mean())
    return float(ece)


def summarize_metrics(y_true: Any, y_pred: Any, num_classes: int = 5) -> dict[str, Any]:
    """Backward-compatible wrapper for the metric bundle."""
    return compute_metrics(y_true, y_pred, num_classes=num_classes)


if __name__ == "__main__":
    y_true = np.array([0, 0, 1, 3, 4])
    y_pred = np.array([0, 1, 1, 2, 4])
    probs = np.eye(5)[y_pred] * 0.8 + 0.05
    metrics = compute_metrics(y_true, y_pred, num_classes=5)
    metrics["ece"] = expected_calibration_error(probs, y_true)
    print({k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in metrics.items()})
