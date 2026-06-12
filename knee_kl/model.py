"""Model definition for knee KL grading."""

from __future__ import annotations

from typing import Any

import timm
import torch
import torch.nn.functional as F
from torch import nn

from knee_kl.attention import build_attention


class L2Normalize(nn.Module):
    """Normalize projection embeddings onto the unit hypersphere."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2, dim=1)


def build_backbone(backbone_name: str, pretrained: bool = True) -> tuple[nn.Module, int]:
    """Build a timm backbone and return it with its feature dimension."""
    backbone = timm.create_model(
        backbone_name,
        pretrained=pretrained,
        num_classes=0,
        global_pool="",
    )
    return backbone, int(backbone.num_features)


def build_classifier_head(in_features: int, num_classes: int) -> nn.Module:
    """Build the KL classification head."""
    return nn.Linear(in_features, num_classes)


def build_projection_head(in_features: int, proj_dim: int) -> nn.Module:
    """Build the contrastive projection head."""
    return nn.Sequential(
        nn.Linear(in_features, in_features),
        nn.ReLU(inplace=True),
        nn.Linear(in_features, proj_dim),
        L2Normalize(),
    )


class KneeKLNet(nn.Module):
    """Backbone + attention + classifier/projection heads."""

    def __init__(self, config: Any, pretrained: bool = True) -> None:
        super().__init__()
        self.cfg = config
        self.backbone, self.feat_dim = build_backbone(config.backbone, pretrained=pretrained)
        self.attention = build_attention(config.attention, self.feat_dim)
        self.classifier = build_classifier_head(self.feat_dim, config.num_classes)
        self.projection = (
            build_projection_head(self.feat_dim, config.proj_dim)
            if getattr(config, "use_contrastive", False)
            else None
        )

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        fmap = self.backbone.forward_features(x)
        if fmap.ndim != 4:
            raise RuntimeError(f"Expected 4D feature map from backbone, got shape {tuple(fmap.shape)}")

        if fmap.shape[1] == self.feat_dim:
            pass
        elif fmap.shape[-1] == self.feat_dim:
            fmap = fmap.permute(0, 3, 1, 2).contiguous()
        else:
            raise RuntimeError(
                f"Cannot infer feature layout for shape {tuple(fmap.shape)} "
                f"with feat_dim={self.feat_dim}"
            )

        assert fmap.shape[1] == self.feat_dim
        return fmap

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        fmap = self._forward_features(x)
        fmap = self.attention(fmap)
        # Grad-CAM is mainly for ConvNeXt; Swin fmap is only interface-compatible because token-grid CAM is weaker.
        feat = F.adaptive_avg_pool2d(fmap, output_size=1).flatten(1)
        logits = self.classifier(feat)
        embed = self.projection(feat) if self.projection is not None else None
        return logits, embed, fmap


def create_model(config: Any, pretrained: bool = True) -> KneeKLNet:
    """Create the complete KL grading model."""
    return KneeKLNet(config, pretrained=pretrained)


def forward_model(model: nn.Module, images: torch.Tensor) -> Any:
    """Run the project-wide model forward contract."""
    return model(images)


if __name__ == "__main__":
    from knee_kl.config import Config

    torch.manual_seed(42)
    backbones = ["convnext_tiny", "swin_tiny_patch4_window7_224"]

    with torch.no_grad():
        for backbone_name in backbones:
            cfg = Config(backbone=backbone_name, attention="coord", use_contrastive=True)
            model = KneeKLNet(cfg, pretrained=True).eval()
            x = torch.randn(2, 3, 224, 224)
            logits, embed, fmap = model(x)
            print(
                f"backbone={backbone_name} "
                f"feat_dim={model.feat_dim} "
                f"fmap.shape={list(fmap.shape)} "
                f"logits.shape={list(logits.shape)} "
                f"embed.shape={list(embed.shape) if embed is not None else None}"
            )

        cfg = Config(backbone="convnext_tiny", attention="none", use_contrastive=False)
        model = KneeKLNet(cfg, pretrained=True).eval()
        x = torch.randn(2, 3, 224, 224)
        logits, embed, fmap = model(x)
        print(
            f"attention=none use_contrastive=False "
            f"feat_dim={model.feat_dim} "
            f"fmap.shape={list(fmap.shape)} "
            f"logits.shape={list(logits.shape)} "
            f"embed={embed}"
        )
