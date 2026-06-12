"""Temperature scaling calibration utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from knee_kl.metrics import expected_calibration_error


class TemperatureScaler(nn.Module):
    """Single-parameter temperature scaler for softmax-style class logits.

    Temperature scaling is intended for softmax losses such as cdw_ce, ce, sord,
    and gaussian_soft. CORN uses num_classes-1 cumulative logits, so this scaler
    is not directly applicable to CORN calibration.
    """

    def __init__(self, init_temperature: float = 1.5) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(torch.tensor(float(init_temperature)).log())

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp_min(1e-6)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(self, val_logits: torch.Tensor, val_labels: torch.Tensor) -> float:
        """Fit T on validation logits by minimizing NLL with LBFGS."""
        self.train()
        val_logits = val_logits.detach()
        val_labels = val_labels.detach().long()
        optimizer = torch.optim.LBFGS([self.log_temperature], lr=0.1, max_iter=50)

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = F.cross_entropy(self(val_logits), val_labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return float(self.temperature.detach().cpu())


def _reliability_bins(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> dict[str, list[float]]:
    confidences = probs.max(dim=1).values.detach().cpu().numpy()
    preds = probs.argmax(dim=1).detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    correct = (preds == labels_np).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    confs: list[float] = []
    accs: list[float] = []
    counts: list[int] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (confidences >= lo) & ((confidences <= hi) if i == n_bins - 1 else (confidences < hi))
        counts.append(int(mask.sum()))
        confs.append(float(confidences[mask].mean()) if mask.any() else 0.0)
        accs.append(float(correct[mask].mean()) if mask.any() else 0.0)
    return {"confidence": confs, "accuracy": accs, "count": counts}


def compare_ece(
    eval_logits: torch.Tensor,
    eval_labels: torch.Tensor,
    temperature: float | torch.Tensor,
    n_bins: int = 15,
) -> tuple[dict[str, float], dict[str, dict[str, list[float]]]]:
    """Compare ECE before/after scaling on independent evaluation data.

    Fit T on validation data, but evaluate ECE on a separate set such as test;
    evaluating on the fit data can report overfit, fake calibration gains.
    """
    temp = torch.as_tensor(temperature, dtype=eval_logits.dtype, device=eval_logits.device).clamp_min(1e-6)
    eval_labels = eval_labels.long()
    probs_before = F.softmax(eval_logits, dim=1)
    probs_after = F.softmax(eval_logits / temp, dim=1)
    summary = {
        "ece_before": expected_calibration_error(probs_before, eval_labels, n_bins=n_bins),
        "ece_after": expected_calibration_error(probs_after, eval_labels, n_bins=n_bins),
        "temperature": float(temp.detach().cpu()),
    }
    bins = {
        "before": _reliability_bins(probs_before, eval_labels, n_bins=n_bins),
        "after": _reliability_bins(probs_after, eval_labels, n_bins=n_bins),
    }
    return summary, bins


def fit_temperature(logits: Any, labels: Any) -> TemperatureScaler:
    """Backward-compatible wrapper returning a fitted scaler."""
    scaler = TemperatureScaler()
    scaler.fit(torch.as_tensor(logits).float(), torch.as_tensor(labels).long())
    return scaler


def apply_temperature(logits: Any, temperature: Any) -> torch.Tensor:
    """Apply temperature scaling to logits."""
    logits_t = torch.as_tensor(logits).float()
    if isinstance(temperature, TemperatureScaler):
        return temperature(logits_t)
    return logits_t / torch.as_tensor(temperature, dtype=logits_t.dtype).clamp_min(1e-6)


def evaluate_calibration(probs: Any, labels: Any, n_bins: int = 15) -> dict:
    """Evaluate ECE for already computed probabilities."""
    return {"ece": expected_calibration_error(probs, labels, n_bins=n_bins)}


if __name__ == "__main__":
    torch.manual_seed(3)
    num_classes = 5
    fit_labels = torch.randint(0, num_classes, (200,))
    eval_labels = torch.randint(0, num_classes, (200,))

    def make_overconfident(labels: torch.Tensor) -> torch.Tensor:
        logits = torch.randn(labels.numel(), num_classes) * 0.2 - 2.0
        pred = labels.clone()
        wrong = torch.arange(labels.numel()) % 4 == 0
        pred[wrong] = (pred[wrong] + 1) % num_classes
        logits[torch.arange(labels.numel()), pred] = 8.0
        return logits

    fit_logits = make_overconfident(fit_labels)
    eval_logits = make_overconfident(eval_labels)
    scaler = TemperatureScaler()
    temperature = scaler.fit(fit_logits, fit_labels)
    summary, bins = compare_ece(eval_logits, eval_labels, temperature)
    print(
        f"temperature={summary['temperature']:.4f} "
        f"ece_before={summary['ece_before']:.4f} "
        f"ece_after={summary['ece_after']:.4f} "
        f"improved={summary['ece_after'] <= summary['ece_before']} "
        f"bins={len(bins['before']['confidence'])}"
    )
