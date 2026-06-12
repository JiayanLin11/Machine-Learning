"""Checkpoint loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from knee_kl.model import KneeKLNet, create_model


def load_model_from_checkpoint(
    ckpt_path: str | Path,
    cfg: Any,
) -> KneeKLNet:
    """Rebuild a model from config and load a saved state dict."""
    model = create_model(cfg, pretrained=False)
    if cfg.loss_type.lower() == "corn":
        model.classifier = nn.Linear(model.feat_dim, cfg.num_classes - 1)

    state_dict = torch.load(Path(ckpt_path), map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model
