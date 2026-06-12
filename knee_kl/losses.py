"""Loss functions for knee KL grading."""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn.functional as F
from coral_pytorch.losses import corn_loss
from torch import nn


class CDWCE(nn.Module):
    """Class Distance Weighted Cross-Entropy.

    For sample target c and softmax probabilities p_i:
        L = - sum_{i != c} |i - c|^alpha * log(1 - p_i)

    The true class is explicitly masked out to avoid the 0 * log(1 - p_c)
    ambiguity when p_c is close to one.
    """

    def __init__(
        self,
        num_classes: int,
        power: float = 3.0,
        class_weights: torch.Tensor | None = None,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.power = float(power)
        self.eps = float(eps)
        if class_weights is None:
            self.register_buffer("class_weights", None)
        else:
            self.register_buffer("class_weights", class_weights.float())

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.long()
        probs = F.softmax(logits, dim=1).clamp(min=self.eps, max=1.0 - self.eps)
        classes = torch.arange(self.num_classes, device=logits.device).unsqueeze(0)
        target_col = target.unsqueeze(1)

        distances = (classes - target_col).abs().float().pow(self.power)
        non_target = classes != target_col
        losses = -torch.log1p(-probs) * distances * non_target.float()
        sample_loss = losses.sum(dim=1)

        if self.class_weights is not None:
            weights = self.class_weights.to(device=logits.device, dtype=sample_loss.dtype)
            sample_loss = sample_loss * weights[target]

        return sample_loss.mean()


class OrdinalSupConLoss(nn.Module):
    """Ordinal-aware supervised contrastive loss.

    Positives are only same-label samples. The denominator follows standard
    SupCon and includes every non-self sample, positives and negatives. Positive
    denominator weights are 1; only negatives are weighted by |y_i - y_j|^beta
    so farther ordinal classes are pushed away more strongly.
    """

    def __init__(self, temperature: float = 0.1, beta: float = 1.0, eps: float = 1e-12) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.beta = float(beta)
        self.eps = float(eps)

    def forward(self, embeddings: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2D [B, D], got {tuple(embeddings.shape)}")

        batch_size = embeddings.shape[0]
        if batch_size <= 1:
            return embeddings.sum() * 0.0

        target = target.long()
        z = F.normalize(embeddings, p=2, dim=1)
        sim = torch.matmul(z, z.t()) / self.temperature

        eye = torch.eye(batch_size, device=embeddings.device, dtype=torch.bool)
        same_label = target.unsqueeze(0) == target.unsqueeze(1)
        positive_mask = same_label & ~eye
        non_self = ~eye

        distances = (target.unsqueeze(0) - target.unsqueeze(1)).abs().float()
        ordinal_weights = torch.ones_like(distances)
        negative_mask = (~same_label) & non_self
        ordinal_weights[negative_mask] = distances[negative_mask].clamp_min(1.0).pow(self.beta)

        weighted_sim = sim + ordinal_weights.clamp_min(self.eps).log()
        weighted_sim = weighted_sim.masked_fill(eye, torch.finfo(weighted_sim.dtype).min)
        log_den = torch.logsumexp(weighted_sim, dim=1)

        positive_counts = positive_mask.sum(dim=1)
        valid_anchor = positive_counts > 0
        if not torch.any(valid_anchor):
            return embeddings.sum() * 0.0

        log_prob = sim - log_den.unsqueeze(1)
        anchor_loss = -(log_prob * positive_mask.float()).sum(dim=1) / positive_counts.clamp_min(1)
        return anchor_loss[valid_anchor].mean()


class MSEOrdinalLoss(nn.Module):
    """Regression baseline for ordinal KL labels.

    If logits has one output, regress that scalar to the class index. If logits
    has C outputs, regress the softmax expected class index to the target.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = int(num_classes)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_float = target.float()
        if logits.shape[1] == 1:
            pred = logits.squeeze(1)
        else:
            classes = torch.arange(logits.shape[1], device=logits.device, dtype=logits.dtype)
            pred = (F.softmax(logits, dim=1) * classes).sum(dim=1)
        return F.mse_loss(pred, target_float)


class CornLossWrapper(nn.Module):
    """CORN loss wrapper.

    CORN expects logits with shape [B, num_classes - 1]. The output-layer change
    is handled by model integration; this wrapper only computes the loss.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = int(num_classes)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return corn_loss(logits, target.long(), self.num_classes)


class SORDLoss(nn.Module):
    """Soft Ordinal Distribution loss with distance-decayed soft labels."""

    def __init__(self, num_classes: int, temperature: float = 1.0) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.temperature = float(temperature)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        classes = torch.arange(self.num_classes, device=logits.device).unsqueeze(0)
        distances = (classes - target.long().unsqueeze(1)).abs().float()
        soft_target = F.softmax(-distances / self.temperature, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        return -(soft_target * log_probs).sum(dim=1).mean()


class GaussianSoftLoss(nn.Module):
    """Gaussian-kernel soft labels plus CE.

    This is the Gaussian-kernel counterpart to SORD's exponential distance
    kernel, useful for comparing kernel shape effects in ordinal learning.
    """

    def __init__(self, num_classes: int, sigma: float = 1.0) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.sigma = float(sigma)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        classes = torch.arange(self.num_classes, device=logits.device).unsqueeze(0)
        distances = (classes - target.long().unsqueeze(1)).float()
        soft_target = F.softmax(-(distances.square()) / (2.0 * self.sigma * self.sigma), dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        return -(soft_target * log_probs).sum(dim=1).mean()


def compute_class_weights(labels: Sequence[int] | torch.Tensor, num_classes: int) -> torch.Tensor:
    """Compute inverse-frequency class weights; missing classes receive 0."""
    labels_tensor = torch.as_tensor(labels, dtype=torch.long)
    counts = torch.bincount(labels_tensor, minlength=num_classes).float()
    missing = counts == 0
    counts = counts.clamp_min(1.0)
    weights = labels_tensor.numel() / (num_classes * counts)
    weights[missing] = 0.0
    present = weights > 0
    if not torch.any(present):
        return weights
    return weights / weights[present].mean()


def build_classification_loss(config: Any, class_weights: torch.Tensor | None = None) -> nn.Module:
    """Build the configured classification/ordinal loss."""
    loss_type = config.loss_type.lower()
    use_class_weight = (
        getattr(config, "use_imbalance", False)
        and getattr(config, "use_class_weight", True)
    )
    weights = class_weights if use_class_weight else None

    if loss_type == "cdw_ce":
        return CDWCE(config.num_classes, power=config.cdw_power, class_weights=weights)
    if loss_type == "ce":
        return nn.CrossEntropyLoss(weight=weights)
    if loss_type == "mse":
        return MSEOrdinalLoss(config.num_classes)
    if loss_type == "corn":
        return CornLossWrapper(config.num_classes)
    if loss_type == "sord":
        return SORDLoss(config.num_classes, temperature=getattr(config, "sord_temperature", 1.0))
    if loss_type == "gaussian_soft":
        return GaussianSoftLoss(config.num_classes, sigma=getattr(config, "gaussian_sigma", 1.0))

    raise ValueError(f"Unsupported loss_type: {config.loss_type!r}")


def total_loss(
    logits: torch.Tensor,
    embed: torch.Tensor | None,
    target: torch.Tensor,
    cls_loss_fn: nn.Module,
    contrastive_fn: nn.Module | None,
    config: Any,
) -> torch.Tensor:
    """Combine classification loss and optional contrastive loss."""
    cls_loss = cls_loss_fn(logits, target)
    if not getattr(config, "use_contrastive", False) or embed is None or contrastive_fn is None:
        return cls_loss
    con_loss = contrastive_fn(embed, target)
    return cls_loss + float(config.contrastive_lambda) * con_loss


# Backward-compatible wrappers for the original placeholder API.
def build_loss(config: Any, class_weights: Any = None) -> nn.Module:
    return build_classification_loss(config, class_weights=class_weights)


def compute_cdw_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    cdw_power: float,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    return CDWCE(logits.shape[1], power=cdw_power, class_weights=class_weights)(logits, targets)


def compute_contrastive_loss(embeddings: torch.Tensor, targets: torch.Tensor, temperature: float) -> torch.Tensor:
    return OrdinalSupConLoss(temperature=temperature)(embeddings, targets)


def combine_losses(main_loss: torch.Tensor, contrastive_loss: torch.Tensor, contrastive_lambda: float) -> torch.Tensor:
    return main_loss + float(contrastive_lambda) * contrastive_loss


def _grad_nonzero(tensor: torch.Tensor) -> bool:
    return tensor.grad is not None and bool(torch.any(tensor.grad.abs() > 0).item())


def _backward_check(name: str, loss: torch.Tensor, tensors: list[torch.Tensor]) -> None:
    for tensor in tensors:
        if tensor.grad is not None:
            tensor.grad.zero_()
    loss.backward()
    grad_nonzero = all(_grad_nonzero(tensor) for tensor in tensors)
    print(f"{name}: loss={loss.item():.6f} backward_ok=True grad_nonzero={grad_nonzero}")


if __name__ == "__main__":
    from knee_kl.config import Config

    torch.manual_seed(7)
    cfg = Config(batch_size=8, num_classes=5, contrastive_temp=0.1, contrastive_beta=1.0)
    labels = torch.tensor([0, 1, 1, 2, 3, 4, 4, 4])
    weights = compute_class_weights(labels, cfg.num_classes)
    missing_weights = compute_class_weights(torch.tensor([0, 0, 1, 1, 3, 3]), cfg.num_classes)
    print(f"class_weights_all_present={weights.tolist()}")
    print(f"class_weights_missing_classes={missing_weights.tolist()}")

    for loss_type in ("cdw_ce", "ce", "corn", "sord", "gaussian_soft"):
        cfg.loss_type = loss_type
        loss_fn = build_classification_loss(cfg, class_weights=weights)
        out_dim = cfg.num_classes - 1 if loss_type == "corn" else cfg.num_classes
        logits = torch.randn(labels.numel(), out_dim, requires_grad=True)
        loss = loss_fn(logits, labels)
        _backward_check(loss_type, loss, [logits])

    cfg.loss_type = "mse"
    mse_fn = build_classification_loss(cfg, class_weights=weights)
    mse_logits_scalar = torch.randn(labels.numel(), 1, requires_grad=True)
    mse_loss_scalar = mse_fn(mse_logits_scalar, labels)
    _backward_check("mse_scalar", mse_loss_scalar, [mse_logits_scalar])
    mse_logits_expected = torch.randn(labels.numel(), cfg.num_classes, requires_grad=True)
    mse_loss_expected = mse_fn(mse_logits_expected, labels)
    _backward_check("mse_expectation", mse_loss_expected, [mse_logits_expected])

    cdw = CDWCE(num_classes=5, power=3.0)
    true_label = torch.tensor([2])
    near_wrong = torch.tensor([[-8.0, -8.0, -8.0, 8.0, -8.0]], requires_grad=True)
    far_wrong = torch.tensor([[8.0, -8.0, -8.0, -8.0, -8.0]], requires_grad=True)
    correct = torch.tensor([[-8.0, -8.0, 8.0, -8.0, -8.0]], requires_grad=True)
    near_loss = cdw(near_wrong, true_label)
    far_loss = cdw(far_wrong, true_label)
    correct_loss = cdw(correct, true_label)
    print(
        "cdw_sanity: "
        f"correct_loss={correct_loss.item():.8f} "
        f"near_wrong_loss={near_loss.item():.6f} "
        f"far_wrong_loss={far_loss.item():.6f} "
        f"far_gt_near={far_loss.item() > near_loss.item()} "
        f"correct_near_zero={correct_loss.item() < 1e-3}"
    )

    con = OrdinalSupConLoss(temperature=cfg.contrastive_temp, beta=cfg.contrastive_beta)
    normal_labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    normal_embeddings = torch.randn(normal_labels.numel(), 128, requires_grad=True)
    normal_loss = con(normal_embeddings, normal_labels)
    normal_finite = torch.isfinite(normal_loss).item()
    normal_loss.backward()
    normal_grad_nonzero = _grad_nonzero(normal_embeddings)
    print(
        f"ordinal_supcon_normal: loss={normal_loss.item():.6f} "
        f"positive_finite={normal_loss.item() > 0 and normal_finite} "
        f"backward_ok=True grad_nonzero={normal_grad_nonzero}"
    )

    sparse_labels = torch.tensor([0, 1, 1, 2, 3, 4])
    sparse_embeddings = torch.randn(sparse_labels.numel(), 128, requires_grad=True)
    sparse_loss = con(sparse_embeddings, sparse_labels)
    sparse_finite = torch.isfinite(sparse_loss).item()
    sparse_loss.backward()
    print(f"ordinal_supcon_sparse: loss={sparse_loss.item():.6f} finite={sparse_finite} backward_ok=True")

    unique_labels = torch.tensor([0, 1, 2, 3, 4])
    unique_embeddings = torch.randn(unique_labels.numel(), 128, requires_grad=True)
    unique_loss = con(unique_embeddings, unique_labels)
    unique_finite = torch.isfinite(unique_loss).item()
    unique_loss.backward()
    print(
        f"ordinal_supcon_all_unique: loss={unique_loss.item():.6f} "
        f"finite={unique_finite} backward_ok=True"
    )

    cfg.loss_type = "cdw_ce"
    logits = torch.randn(labels.numel(), cfg.num_classes, requires_grad=True)
    embeddings = torch.randn(labels.numel(), cfg.proj_dim, requires_grad=True)
    cls_loss_fn = build_classification_loss(cfg)
    total = total_loss(logits, embeddings, labels, cls_loss_fn, con, cfg)
    _backward_check("total_loss_contrastive_on", total, [logits, embeddings])

    cfg.use_contrastive = False
    logits = torch.randn(labels.numel(), cfg.num_classes, requires_grad=True)
    total = total_loss(logits, None, labels, cls_loss_fn, con, cfg)
    _backward_check("total_loss_contrastive_off", total, [logits])
