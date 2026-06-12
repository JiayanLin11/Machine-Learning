"""Attention modules for feature maps shaped as [B, C, H, W]."""

from __future__ import annotations

import torch
from torch import nn


class CoordinateAttention(nn.Module):
    """Coordinate Attention from Hou et al. 2021."""

    def __init__(self, channels: int, reduction: int = 32) -> None:
        super().__init__()
        hidden_channels = max(8, channels // reduction)
        self.conv1 = nn.Conv2d(channels, hidden_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.act = nn.Hardswish()
        self.conv_h = nn.Conv2d(hidden_channels, channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(hidden_channels, channels, kernel_size=1, stride=1, padding=0)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        _b, _c, height, width = x.shape

        x_h = x.mean(dim=3, keepdim=True)
        x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)

        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [height, width], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        attended = identity * a_h * a_w
        return identity + self.gamma * (attended - identity)


class SEAttention(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden_channels = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


class ChannelAttention(nn.Module):
    """CBAM channel attention."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden_channels = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.mlp(torch.mean(x, dim=(2, 3), keepdim=True))
        max_pool = self.mlp(torch.amax(x, dim=(2, 3), keepdim=True))
        return self.sigmoid(avg + max_pool)


class SpatialAttention(nn.Module):
    """CBAM spatial attention."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        max_pool = torch.amax(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg, max_pool], dim=1)))


class CBAMAttention(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction=reduction)
        self.spatial_attention = SpatialAttention(kernel_size=7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_attention(x)
        return x * self.spatial_attention(x)


class ZPool(nn.Module):
    """Concatenate max and mean projections along the channel axis."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [torch.amax(x, dim=1, keepdim=True), torch.mean(x, dim=1, keepdim=True)],
            dim=1,
        )


class AttentionGate(nn.Module):
    """Triplet Attention gate over the last two dimensions after Z-pooling."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.compress = ZPool()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.sigmoid(self.conv(self.compress(x)))
        return x * scale


class TripletAttention(nn.Module):
    """Triplet Attention with C-H, C-W, and H-W cross-dimension branches."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.cw_gate = AttentionGate(kernel_size=kernel_size)
        self.hc_gate = AttentionGate(kernel_size=kernel_size)
        self.hw_gate = AttentionGate(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_cw = self.cw_gate(x.permute(0, 2, 1, 3).contiguous()).permute(0, 2, 1, 3).contiguous()
        x_hc = self.hc_gate(x.permute(0, 3, 2, 1).contiguous()).permute(0, 3, 2, 1).contiguous()
        x_hw = self.hw_gate(x)
        return (x_cw + x_hc + x_hw) / 3.0


def build_attention(name: str, channels: int, reduction: int | None = None) -> nn.Module:
    """Build an attention module with a uniform [B, C, H, W] interface."""
    attention_name = name.lower()
    if attention_name == "none":
        return nn.Identity()
    if attention_name == "coord":
        return CoordinateAttention(channels, reduction=32 if reduction is None else reduction)
    if attention_name == "se":
        return SEAttention(channels, reduction=16 if reduction is None else reduction)
    if attention_name == "cbam":
        return CBAMAttention(channels, reduction=16 if reduction is None else reduction)
    if attention_name == "triplet":
        return TripletAttention()
    raise ValueError(f"Unsupported attention type: {name!r}")


def apply_attention(features: torch.Tensor, attention_module: nn.Module) -> torch.Tensor:
    """Apply an attention module to a feature map."""
    return attention_module(features)


if __name__ == "__main__":
    torch.manual_seed(42)

    for attention_name in ("none", "coord", "se", "cbam", "triplet"):
        module = build_attention(attention_name, channels=32)
        x = torch.randn(2, 32, 7, 7)
        y = module(x)
        print(f"{attention_name}: {list(y.shape)}")

    for backbone_name, channels, height, width in (
        ("convnext_tiny", 768, 7, 7),
        ("swin_tiny_patch4_window7_224", 768, 7, 7),
    ):
        module = build_attention("coord", channels=channels)
        x = torch.randn(2, channels, height, width)
        y = module(x)
        print(f"coord backbone={backbone_name}: {list(y.shape)}")

    module = build_attention("coord", channels=32)
    assert isinstance(module, CoordinateAttention)
    print(f"coord gamma init: {module.gamma.item():.6f}")

    x = torch.randn(2, 32, 7, 7)
    y = module(x)
    max_abs_diff = (y - x).abs().max().item()
    print(f"coord gamma=0 allclose: {torch.allclose(y, x, atol=1e-6)}")
    print(f"coord gamma=0 max_abs_diff: {max_abs_diff:.9f}")

    y.sum().backward()
    gamma_grad = module.gamma.grad
    print(f"coord gamma grad: {gamma_grad.item() if gamma_grad is not None else None}")
    print(f"coord gamma grad nonzero: {gamma_grad is not None and bool(gamma_grad.abs().max() > 0)}")
