"""Visualization helpers for Grad-CAM and t-SNE."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from torch import nn


class _LogitsOnly(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _embed, _fmap = self.model(x)
        return logits


def _reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 4 and tensor.shape[1] <= 16 and tensor.shape[-1] > 16:
        return tensor.permute(0, 3, 1, 2).contiguous()
    if tensor.ndim == 3:
        b, n, c = tensor.shape
        side = int(n ** 0.5)
        if side * side == n:
            return tensor.reshape(b, side, side, c).permute(0, 3, 1, 2).contiguous()
    return tensor


def _last_conv_module(module: nn.Module) -> nn.Module | None:
    last = None
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            last = child
    return last


def _default_target_layer(model: nn.Module, cfg: Any) -> nn.Module:
    if getattr(cfg, "attention", "none").lower() != "none" and not isinstance(model.attention, nn.Identity):
        return model.attention
    # For attention=none, use the last convolutional block/stage we can locate in the backbone.
    layer = _last_conv_module(model.backbone)
    return layer if layer is not None else model.backbone


def gradcam(model: nn.Module, image: torch.Tensor, cfg: Any, target_layer: nn.Module | None = None) -> np.ndarray:
    """Generate a Grad-CAM heatmap array.

    Grad-CAM is mainly for ConvNeXt. Swin transformer token-grid CAM has weaker
    interpretability, but this path keeps the visualization interface usable.
    """
    model.eval()
    device = next(model.parameters()).device
    image = image.to(device)
    if image.ndim == 3:
        image = image.unsqueeze(0)
    target_layer = target_layer or _default_target_layer(model, cfg)
    wrapper = _LogitsOnly(model)
    with torch.no_grad():
        target_idx = int(wrapper(image).argmax(dim=1)[0].item())
    with GradCAM(model=wrapper, target_layers=[target_layer], reshape_transform=_reshape_transform) as cam:
        grayscale_cam = cam(input_tensor=image, targets=[ClassifierOutputTarget(target_idx)])
    return grayscale_cam[0]


def tsne_silhouette(features: Any, labels: Any) -> tuple[np.ndarray, float]:
    """Return t-SNE 2D coordinates and silhouette computed on original features."""
    feats = np.asarray(features.detach().cpu() if hasattr(features, "detach") else features, dtype=float)
    labs = np.asarray(labels.detach().cpu() if hasattr(labels, "detach") else labels, dtype=int)
    perplexity = max(2, min(30, feats.shape[0] - 1))
    coords = TSNE(n_components=2, perplexity=perplexity, init="random", learning_rate="auto", random_state=42).fit_transform(feats)
    unique = np.unique(labs)
    sil = float("nan")
    if unique.size > 1 and unique.size < feats.shape[0]:
        sil = float(silhouette_score(feats, labs))
    return coords, sil


def generate_grad_cam(model: Any, images: Any, target_layer: Any, targets: Any = None) -> Any:
    """Backward-compatible wrapper."""
    cfg = getattr(model, "cfg", None)
    return gradcam(model, images, cfg, target_layer=target_layer)


def extract_embeddings(model: Any, dataloader: Any) -> Any:
    """Extract pooled feature vectors, labels and no extra metadata."""
    model.eval()
    device = next(model.parameters()).device
    feats, labels = [], []
    with torch.no_grad():
        for images, target in dataloader:
            _logits, _embed, fmap = model(images.to(device))
            feats.append(F.adaptive_avg_pool2d(fmap, 1).flatten(1).cpu())
            labels.append(target.cpu())
    return torch.cat(feats), torch.cat(labels), None


def run_tsne(embeddings: Any, labels: Any, config: Any) -> Any:
    """Backward-compatible t-SNE wrapper."""
    return tsne_silhouette(embeddings, labels)


def save_visualization(figure: Any, output_path: str) -> None:
    """Save a matplotlib-like figure."""
    figure.savefig(output_path, bbox_inches="tight")


if __name__ == "__main__":
    from knee_kl.config import Config
    from knee_kl.model import KneeKLNet

    torch.manual_seed(9)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = torch.randn(1, 3, 64, 64)

    for attention in ("coord", "none"):
        cfg = Config(img_size=64, backbone="convnext_tiny", attention=attention)
        model = KneeKLNet(cfg).to(device).eval()
        heatmap = gradcam(model, image, cfg)
        print(f"gradcam_{attention}_shape={list(heatmap.shape)} min={float(np.min(heatmap)):.4f} max={float(np.max(heatmap)):.4f}")

    cfg_swin = Config(img_size=224, backbone="swin_tiny_patch4_window7_224", attention="coord")
    model_swin = KneeKLNet(cfg_swin).to(device).eval()
    swin_image = torch.randn(1, 3, 224, 224)
    heatmap = gradcam(model_swin, swin_image, cfg_swin)
    print(f"gradcam_swin_shape={list(heatmap.shape)} min={float(np.min(heatmap)):.4f} max={float(np.max(heatmap)):.4f}")

    features = torch.randn(20, 16)
    labels = torch.arange(20) % 5
    coords, sil = tsne_silhouette(features, labels)
    print(f"tsne_shape={list(coords.shape)} silhouette={sil:.4f}")
