"""FFE classifier with a GAP vs second-order pooling switch."""

from __future__ import annotations

import timm
import torch
import torch.nn as nn

from src.secondorder import SecondOrderPool


class PooledHierarchicalClassifier(nn.Module):
    """Return the same heads as HierarchicalClassifier with configurable pooling."""

    def __init__(
        self,
        backbone_tag: str,
        num_classes: int,
        num_species: int,
        pooling: str = "gap",
        proj_dim: int = 128,
        pretrained: bool = True,
        dropout: float = 0.3,
        drop_path: float = 0.1,
    ):
        super().__init__()
        if pooling not in {"gap", "raw_bilinear", "gap_raw_bilinear", "centered_cov"}:
            raise ValueError(f"Unsupported pooling: {pooling!r}")
        self.pooling = pooling
        self.backbone = timm.create_model(
            backbone_tag,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
            drop_rate=dropout,
            drop_path_rate=drop_path,
        )
        channels = self.backbone.num_features
        self.second_order = None
        if pooling in {"raw_bilinear", "gap_raw_bilinear", "centered_cov"}:
            mode = "centered_cov" if pooling == "centered_cov" else "raw_bilinear"
            self.second_order = SecondOrderPool(channels, proj_dim=proj_dim, mode=mode)
            feature_dim = self.second_order.out_dim
            if pooling == "gap_raw_bilinear":
                feature_dim += channels
        else:
            feature_dim = channels

        self.head_drop = nn.Dropout(dropout)
        self.class_head = nn.Linear(feature_dim, num_classes)
        self.species_head = nn.Linear(feature_dim, num_species)
        self.freshness_head = nn.Linear(feature_dim, 2)

    def _gap(self, feat: torch.Tensor) -> torch.Tensor:
        channels = self.backbone.num_features
        if feat.dim() == 4:
            if feat.shape[1] == channels:
                return feat.mean(dim=(2, 3))
            if feat.shape[-1] == channels:
                return feat.mean(dim=(1, 2))
        if feat.dim() == 3:
            if feat.shape[-1] == channels:
                return feat.mean(dim=1)
            if feat.shape[1] == channels:
                return feat.mean(dim=2)
        raise ValueError(f"Cannot GAP feature map with shape {tuple(feat.shape)}")

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        fmap = self.backbone.forward_features(x)
        gap = self._gap(fmap)
        if self.pooling == "gap":
            return gap
        assert self.second_order is not None
        second = self.second_order(fmap)
        if self.pooling == "gap_raw_bilinear":
            return torch.cat([gap.float(), second], dim=1)
        return second

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.head_drop(self.extract_features(x))
        return {
            "class": self.class_head(feat),
            "species": self.species_head(feat),
            "freshness": self.freshness_head(feat),
        }
