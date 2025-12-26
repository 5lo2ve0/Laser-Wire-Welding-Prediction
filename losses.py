# losses.py
# Auxiliary losses used to improve WGAN-GP stability (representative).

from __future__ import annotations
import torch
import torch.nn.functional as F


def consistency_loss(fake_sample: torch.Tensor, real_sample: torch.Tensor) -> torch.Tensor:
    """MSE between generated and real samples (soft constraint)."""
    return F.mse_loss(fake_sample, real_sample)


def feature_loss(fake_feature: torch.Tensor, real_feature: torch.Tensor) -> torch.Tensor:
    """MSE between critic internal features (feature matching)."""
    return F.mse_loss(fake_feature, real_feature)


def center_loss(fake_sample: torch.Tensor, real_sample: torch.Tensor) -> torch.Tensor:
    """L2 distance between centroids of fake and real samples."""
    return torch.norm(fake_sample.mean(dim=0) - real_sample.mean(dim=0), p=2)
