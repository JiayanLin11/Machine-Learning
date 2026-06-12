"""Test-time augmentation and threshold optimization."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score
from torchvision.transforms import functional as TF

from knee_kl.metrics import compute_metrics, compute_qwk, expected_calibration_error, probabilities_from_logits


def build_tta_transforms(config: Any) -> list[str]:
    """Return the mild TTA operations used at inference."""
    return ["identity", "horizontal_flip", "small_rotation"]


def _augment_batch(images: torch.Tensor, aug_idx: int, n_aug: int) -> torch.Tensor:
    if aug_idx == 0:
        return images
    out = images
    if aug_idx % 2 == 1:
        out = torch.flip(out, dims=[-1])
    angle = -10.0 + 20.0 * (aug_idx / max(1, n_aug - 1))
    if abs(angle) > 1e-6:
        out = TF.rotate(out, angle=float(angle), interpolation=TF.InterpolationMode.BILINEAR)
    return out


def predict_with_tta(model: Any, images: torch.Tensor, device: torch.device | str, cfg: Any, n_aug: int = 5) -> torch.Tensor:
    """Average probabilities over horizontal flips and small rotations; no vertical flips."""
    model.eval()
    device = torch.device(device)
    probs_sum: torch.Tensor | None = None
    with torch.no_grad():
        for aug_idx in range(n_aug):
            aug_images = _augment_batch(images.to(device), aug_idx, n_aug)
            logits, _embed, _fmap = model(aug_images)
            probs = probabilities_from_logits(logits, cfg)
            probs_sum = probs if probs_sum is None else probs_sum + probs
    return (probs_sum / float(n_aug)).detach().cpu()


def evaluate_with_tta(model: Any, loader: Any, device: torch.device | str, cfg: Any, n_aug: int = 5) -> dict[str, Any]:
    """Evaluate a loader using TTA-averaged probabilities."""
    all_probs: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    for images, labels in loader:
        all_probs.append(predict_with_tta(model, images, device, cfg, n_aug=n_aug))
        all_labels.append(labels.detach().cpu().long())
    probs = torch.cat(all_probs)
    labels = torch.cat(all_labels)
    preds = probs.argmax(dim=1)
    metrics = compute_metrics(labels, preds, num_classes=cfg.num_classes)
    metrics["ece"] = expected_calibration_error(probs, labels)
    return metrics


def apply_thresholds(probs: Any, thresholds: Any) -> np.ndarray:
    """Apply per-class acceptance thresholds; fallback to highest p_i / threshold_i score."""
    probs_np = np.asarray(probs.detach().cpu() if hasattr(probs, "detach") else probs, dtype=float)
    thresholds_np = np.asarray(thresholds, dtype=float)
    preds: list[int] = []
    for row in probs_np:
        accepted = np.where(row >= thresholds_np)[0]
        if accepted.size:
            preds.append(int(accepted[np.argmax(row[accepted])]))
        else:
            preds.append(int(np.argmax(row / np.clip(thresholds_np, 1e-6, None))))
    return np.asarray(preds, dtype=int)


def optimize_thresholds(val_probs: Any, val_labels: Any, num_classes: int = 5) -> dict[str, Any]:
    """Tune per-class probability thresholds on validation data for QWK.

    This is an add-on post-processing experiment. Main results should use plain
    argmax or CORN decoding; report threshold gains together with per-class F1
    before/after to catch damage to minority classes such as KL-1 or KL-4.
    """
    probs = np.asarray(val_probs.detach().cpu() if hasattr(val_probs, "detach") else val_probs, dtype=float)
    labels = np.asarray(val_labels.detach().cpu() if hasattr(val_labels, "detach") else val_labels, dtype=int)
    thresholds = np.full(num_classes, 0.5, dtype=float)
    before_preds = probs.argmax(axis=1)
    before_qwk = compute_qwk(labels, before_preds, num_classes=num_classes)
    best_qwk = compute_qwk(labels, apply_thresholds(probs, thresholds), num_classes=num_classes)
    grid = np.linspace(0.05, 0.95, 19)
    for _ in range(2):
        for cls in range(num_classes):
            best_val = thresholds[cls]
            for candidate in grid:
                trial = thresholds.copy()
                trial[cls] = candidate
                qwk = compute_qwk(labels, apply_thresholds(probs, trial), num_classes=num_classes)
                if qwk > best_qwk:
                    best_qwk = qwk
                    best_val = candidate
            thresholds[cls] = best_val
    after_preds = apply_thresholds(probs, thresholds)
    f1_before = f1_score(labels, before_preds, labels=list(range(num_classes)), average=None, zero_division=0).tolist()
    f1_after = f1_score(labels, after_preds, labels=list(range(num_classes)), average=None, zero_division=0).tolist()
    return {
        "thresholds": thresholds.tolist(),
        "qwk_before": float(before_qwk),
        "qwk_after": float(best_qwk),
        "f1_per_class_before": f1_before,
        "f1_per_class_after": f1_after,
    }


if __name__ == "__main__":
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    from knee_kl.config import Config
    from knee_kl.model import KneeKLNet

    torch.manual_seed(5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images = torch.randn(4, 3, 64, 64)
    labels = torch.arange(4) % 5
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)

    cfg = Config(img_size=64, backbone="convnext_tiny", attention="coord", loss_type="cdw_ce")
    model = KneeKLNet(cfg).to(device).eval()
    probs = predict_with_tta(model, images, device, cfg, n_aug=3)
    print(f"cdw_tta_shape={list(probs.shape)} row_sums={[round(x, 4) for x in probs.sum(dim=1).tolist()]}")
    metrics = evaluate_with_tta(model, loader, device, cfg, n_aug=2)
    print(f"cdw_tta_metrics keys={sorted(metrics.keys())} ece={metrics['ece']:.4f}")

    cfg_corn = Config(img_size=64, backbone="convnext_tiny", attention="coord", loss_type="corn", use_contrastive=False)
    model_corn = KneeKLNet(cfg_corn).to(device).eval()
    model_corn.classifier = nn.Linear(model_corn.feat_dim, cfg_corn.num_classes - 1).to(device)
    corn_probs = predict_with_tta(model_corn, images, device, cfg_corn, n_aug=2)
    print(f"corn_tta_shape={list(corn_probs.shape)} row_sums={[round(x, 4) for x in corn_probs.sum(dim=1).tolist()]}")
    corn_metrics = evaluate_with_tta(model_corn, loader, device, cfg_corn, n_aug=2)
    print(f"corn_tta_metrics keys={sorted(corn_metrics.keys())} ece={corn_metrics['ece']:.4f}")

    val_probs = torch.softmax(torch.randn(30, 5), dim=1)
    val_labels = torch.arange(30) % 5
    result = optimize_thresholds(val_probs, val_labels, num_classes=5)
    print(
        f"thresholds={result['thresholds']} qwk_before={result['qwk_before']:.4f} "
        f"qwk_after={result['qwk_after']:.4f} f1_before={result['f1_per_class_before']} "
        f"f1_after={result['f1_per_class_after']}"
    )
