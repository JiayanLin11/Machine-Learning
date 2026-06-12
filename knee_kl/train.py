"""Training and validation loops for knee KL grading."""

from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from coral_pytorch.dataset import corn_label_from_logits
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from knee_kl.losses import (
    OrdinalSupConLoss,
    build_classification_loss,
    compute_class_weights,
    total_loss,
)
from knee_kl.metrics import compute_metrics, expected_calibration_error, probabilities_from_logits
from knee_kl.model import KneeKLNet


MAXIMIZE_METRICS = {"qwk", "acc", "macro_f1"}
MINIMIZE_METRICS = {"mae", "loss"}


def set_seed(seed: int) -> None:
    """Set Python, NumPy and PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _is_better(metric_name: str, current: float, best: float | None) -> bool:
    if best is None:
        return True
    name = metric_name.lower()
    if name in MINIMIZE_METRICS:
        return current < best
    if name in MAXIMIZE_METRICS:
        return current > best
    raise ValueError(f"Unknown monitor metric direction for {metric_name!r}")


def _move_class_weights(class_weights: torch.Tensor | None, device: torch.device) -> torch.Tensor | None:
    return None if class_weights is None else class_weights.to(device)


def _extract_labels_from_dataset(dataset: Any) -> list[int] | None:
    if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
        base_labels = _extract_labels_from_dataset(dataset.dataset)
        if base_labels is not None:
            return [int(base_labels[index]) for index in dataset.indices]
    if hasattr(dataset, "samples"):
        return [int(sample[1]) for sample in dataset.samples]
    if isinstance(dataset, TensorDataset) and len(dataset.tensors) >= 2:
        return [int(label) for label in dataset.tensors[1].detach().cpu().tolist()]
    return None


def _train_labels_from_loader(loader: DataLoader, device: torch.device) -> torch.Tensor:
    labels = _extract_labels_from_dataset(loader.dataset)
    if labels is not None:
        return torch.as_tensor(labels, dtype=torch.long)

    collected: list[torch.Tensor] = []
    for _images, target in loader:
        collected.append(target.detach().cpu().long())
    if not collected:
        return torch.empty(0, dtype=torch.long)
    return torch.cat(collected)


def _check_drop_last(loader: DataLoader) -> None:
    if not getattr(loader, "drop_last", False):
        raise ValueError("train_loader must be created in build_loaders with drop_last=True")


def _adapt_corn_classifier(model: KneeKLNet, config: Any) -> None:
    if config.loss_type.lower() != "corn":
        return
    model.classifier = nn.Linear(model.feat_dim, config.num_classes - 1)


def build_optimizer(model: nn.Module, config: Any) -> torch.optim.Optimizer:
    """Build AdamW with separate backbone and head learning rates."""
    backbone_param_ids = {id(param) for param in model.backbone.parameters()}
    backbone_params = [param for param in model.parameters() if id(param) in backbone_param_ids and param.requires_grad]
    head_params = [param for param in model.parameters() if id(param) not in backbone_param_ids and param.requires_grad]
    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": config.lr_backbone},
            {"params": head_params, "lr": config.lr_head},
        ]
    )


def build_scheduler(optimizer: torch.optim.Optimizer, config: Any) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    """Build cosine annealing scheduler."""
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)


def _predict_from_logits(logits: torch.Tensor, config: Any) -> torch.Tensor:
    if config.loss_type.lower() == "corn":
        # CORN logits are cumulative binary logits; argmax would not represent an ordinal class.
        return corn_label_from_logits(logits)
    return torch.argmax(logits, dim=1)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Any,
    device: torch.device | str = "cpu",
    contrastive_fn: nn.Module | None = None,
) -> dict[str, float]:
    """Run one training epoch."""
    model.train()
    device = torch.device(device)
    running_loss = 0.0
    n_samples = 0

    for images, target in dataloader:
        images = images.to(device)
        target = target.to(device).long()
        optimizer.zero_grad(set_to_none=True)
        logits, embed, _fmap = model(images)
        loss = total_loss(logits, embed, target, criterion, contrastive_fn, config)
        loss.backward()
        optimizer.step()

        batch_size = target.shape[0]
        running_loss += float(loss.detach().cpu()) * batch_size
        n_samples += batch_size

    return {"loss": running_loss / max(1, n_samples)}


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device | str,
    return_logits: bool = False,
    config: Any | None = None,
) -> dict[str, Any] | tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    """Evaluate a model and optionally return concatenated logits and labels."""
    model.eval()
    device = torch.device(device)
    cfg = config if config is not None else getattr(model, "cfg", None)
    if cfg is None:
        raise ValueError("evaluate requires config or model.cfg")

    logits_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    preds_list: list[torch.Tensor] = []
    probs_list: list[torch.Tensor] = []

    with torch.no_grad():
        for images, target in loader:
            images = images.to(device)
            target = target.to(device).long()
            logits, _embed, _fmap = model(images)
            preds = _predict_from_logits(logits, cfg)
            probs = probabilities_from_logits(logits, cfg)

            logits_list.append(logits.detach().cpu())
            labels_list.append(target.detach().cpu())
            preds_list.append(preds.detach().cpu())
            probs_list.append(probs.detach().cpu())

    logits_all = torch.cat(logits_list) if logits_list else torch.empty(0, cfg.num_classes)
    labels_all = torch.cat(labels_list) if labels_list else torch.empty(0, dtype=torch.long)
    preds_all = torch.cat(preds_list) if preds_list else torch.empty(0, dtype=torch.long)
    probs_all = torch.cat(probs_list) if probs_list else torch.empty(0, cfg.num_classes)

    metrics = compute_metrics(labels_all, preds_all, num_classes=cfg.num_classes)
    metrics["ece"] = expected_calibration_error(probs_all, labels_all)

    if return_logits:
        return metrics, logits_all, labels_all
    return metrics


def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    config: Any,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Run one validation epoch with loss and metrics."""
    model.eval()
    device = torch.device(device)
    running_loss = 0.0
    n_samples = 0

    with torch.no_grad():
        for images, target in dataloader:
            images = images.to(device)
            target = target.to(device).long()
            logits, _embed, _fmap = model(images)
            loss = criterion(logits, target)
            batch_size = target.shape[0]
            running_loss += float(loss.detach().cpu()) * batch_size
            n_samples += batch_size

    metrics = evaluate(model, dataloader, device, config=config)
    # Validation loss is classification loss only, while train loss may include contrastive loss;
    # if monitor_metric is "loss", do not compare it directly to train total_loss.
    metrics["loss"] = running_loss / max(1, n_samples)
    return metrics


def _save_checkpoint(
    model: nn.Module,
    cfg: Any,
    config_name: str,
    fold: int,
) -> tuple[Path, Path]:
    checkpoint_dir = Path(getattr(cfg, "output_dir", "outputs")) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{config_name}_seed{cfg.seed}_fold{fold}"
    checkpoint_path = checkpoint_dir / f"{stem}.pt"
    sidecar_path = checkpoint_dir / f"{stem}.json"

    torch.save(model.state_dict(), checkpoint_path)
    metadata = {
        "config_name": config_name,
        "seed": int(cfg.seed),
        "fold": int(fold),
        "backbone": cfg.backbone,
        "attention": cfg.attention,
        "loss_type": cfg.loss_type,
        "num_classes": int(cfg.num_classes),
        "use_imbalance": bool(getattr(cfg, "use_imbalance", False)),
        "use_oversampling": bool(getattr(cfg, "use_oversampling", True)),
        "use_class_weight": bool(getattr(cfg, "use_class_weight", True)),
        "use_contrastive": bool(getattr(cfg, "use_contrastive", False)),
        "config": asdict(cfg) if is_dataclass(cfg) else dict(vars(cfg)),
    }
    with sidecar_path.open("w") as handle:
        json.dump(metadata, handle, indent=2)
    return checkpoint_path, sidecar_path


def train_one_fold(
    cfg: Any,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device | str,
    config_name: str | None = None,
    fold: int | None = None,
) -> nn.Module:
    """Train one fold and return the best-validation model."""
    set_seed(cfg.seed)
    device = torch.device(device)
    _check_drop_last(train_loader)

    model = KneeKLNet(cfg).to(device)
    _adapt_corn_classifier(model, cfg)
    model.to(device)

    train_labels = _train_labels_from_loader(train_loader, device)
    use_class_weight = (
        getattr(cfg, "use_imbalance", False)
        and getattr(cfg, "use_class_weight", True)
    )
    class_weights = compute_class_weights(train_labels, cfg.num_classes) if use_class_weight else None
    cls_loss_fn = build_classification_loss(cfg, _move_class_weights(class_weights, device))
    contrastive_fn = OrdinalSupConLoss(
        temperature=cfg.contrastive_temp,
        beta=getattr(cfg, "contrastive_beta", 1.0),
    ).to(device)

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    monitor = cfg.monitor_metric.lower()
    best_score: float | None = None
    best_state: dict[str, torch.Tensor] | None = None
    bad_epochs = 0

    for epoch in range(cfg.epochs):
        train_log = train_one_epoch(
            model,
            train_loader,
            cls_loss_fn,
            optimizer,
            cfg,
            device=device,
            contrastive_fn=contrastive_fn,
        )
        val_log = validate_one_epoch(model, val_loader, cls_loss_fn, cfg, device=device)
        scheduler.step()

        current = float(val_log[monitor])
        improved = _is_better(monitor, current, best_score)
        if improved:
            best_score = current
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        print(
            f"epoch {epoch + 1}/{cfg.epochs} "
            f"train_loss={train_log['loss']:.6f} "
            f"val_loss={val_log['loss']:.6f} "
            f"qwk={val_log['qwk']:.4f} "
            f"acc={val_log['acc']:.4f} "
            f"mae={val_log['mae']:.4f} "
            f"macro_f1={val_log['macro_f1']:.4f} "
            f"ece={val_log['ece']:.4f} "
            f"improved={improved}"
        )

        if bad_epochs >= cfg.early_stop_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    if getattr(cfg, "save_checkpoints", False):
        if config_name is None or fold is None:
            raise ValueError(
                "config_name and fold are required when save_checkpoints=True"
            )
        _save_checkpoint(model, cfg, config_name, fold)
    return model


def run_training(config: Any, fold: int) -> dict[str, Any]:
    """Placeholder orchestration entry kept for compatibility."""
    raise NotImplementedError("Use train_one_fold with prepared train/val loaders in this phase.")


def _make_random_loader(
    num_samples: int,
    img_size: int,
    num_classes: int,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
) -> DataLoader:
    generator = torch.Generator().manual_seed(123 + num_samples)
    labels = torch.arange(num_samples) % num_classes
    images = torch.randn(num_samples, 3, img_size, img_size, generator=generator)
    dataset = TensorDataset(images, labels.long())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=0)


def _run_smoke(name: str, cfg: Any, device: torch.device) -> None:
    print(f"--- {name} ---")
    train_loader = _make_random_loader(20, cfg.img_size, cfg.num_classes, cfg.batch_size, True, True)
    val_loader = _make_random_loader(10, cfg.img_size, cfg.num_classes, cfg.batch_size, False, False)
    model = train_one_fold(cfg, train_loader, val_loader, device)
    metrics = evaluate(model, val_loader, device, config=cfg)
    print(
        f"final_metrics keys={sorted(metrics.keys())} "
        f"confusion_shape={list(metrics['confusion'].shape)} "
        f"f1_len={len(metrics['f1_per_class'])}"
    )


if __name__ == "__main__":
    from knee_kl.config import Config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_kwargs = dict(
        img_size=64,
        batch_size=4,
        epochs=2,
        early_stop_patience=2,
        num_workers=0,
        backbone="convnext_tiny",
        attention="coord",
        monitor_metric="qwk",
    )

    cfg_full = Config(**base_kwargs)
    _run_smoke("full cdw_ce + coord + imbalance + contrastive", cfg_full, device)

    cfg_ce = Config(**base_kwargs, loss_type="ce", use_contrastive=False)
    _run_smoke("ce + no contrastive", cfg_ce, device)

    cfg_corn = Config(**base_kwargs, loss_type="corn", use_contrastive=False)
    _run_smoke("corn output adaptation", cfg_corn, device)
