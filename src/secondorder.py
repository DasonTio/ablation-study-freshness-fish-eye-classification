"""Mean-preserving second-order pooling for fish-eye turbidity features.

The primary operator is uncentered bilinear pooling: it keeps global mean
opacity/color information while adding pairwise channel interactions. Centered
covariance is retained as an explicit control because the frozen-feature screen
showed that centering can remove useful FFE freshness signal.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def to_bcm(feat: torch.Tensor, num_features: int) -> torch.Tensor:
    """Convert common timm feature-map layouts to [B, C, M]."""
    if feat.dim() == 3:
        if feat.shape[-1] == num_features:
            return feat.transpose(1, 2).contiguous()
        if feat.shape[1] == num_features:
            return feat.contiguous()
    elif feat.dim() == 4:
        b = feat.shape[0]
        if feat.shape[1] == num_features:
            return feat.reshape(b, num_features, -1).contiguous()
        if feat.shape[-1] == num_features:
            return feat.permute(0, 3, 1, 2).reshape(b, num_features, -1).contiguous()
    raise ValueError(f"Cannot infer channel axis for shape={tuple(feat.shape)} C={num_features}")


def _upper_triangle(mats: torch.Tensor) -> torch.Tensor:
    d = mats.shape[-1]
    idx = torch.triu_indices(d, d, device=mats.device)
    return mats[:, idx[0], idx[1]]


class SecondOrderPool(nn.Module):
    """Projected second-order pooling.

    Modes:
    - raw_bilinear: E[xx^T], mean-preserving B-CNN style pooling.
    - centered_cov: E[(x-mu)(x-mu)^T], diagnostic control.
    """

    def __init__(self, in_channels: int, proj_dim: int | None = 128, mode: str = "raw_bilinear"):
        super().__init__()
        if mode not in {"raw_bilinear", "centered_cov"}:
            raise ValueError(f"Unsupported second-order mode: {mode!r}")
        self.in_channels = in_channels
        self.mode = mode
        if proj_dim is not None and proj_dim < in_channels:
            self.proj = nn.Conv1d(in_channels, proj_dim, kernel_size=1, bias=False)
            self.out_channels = proj_dim
        else:
            self.proj = None
            self.out_channels = in_channels
        self.out_dim = self.out_channels * (self.out_channels + 1) // 2

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = to_bcm(feat, self.in_channels).float()
        if self.proj is not None:
            x = self.proj(x)
        m = x.shape[-1]
        if self.mode == "centered_cov":
            x = x - x.mean(dim=2, keepdim=True)
            denom = max(m - 1, 1)
        else:
            denom = m
        gram = (x @ x.transpose(1, 2)) / denom
        vec = _upper_triangle(gram)
        vec = torch.sign(vec) * torch.sqrt(torch.abs(vec) + 1e-12)
        return F.normalize(vec, dim=1)
