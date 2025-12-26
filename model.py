# models.py
# Core model definitions (representative implementation aligned with the study code).
# This file contains model architectures only (no training/evaluation pipeline).

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# -----------------------------
# WGAN-GP components
# -----------------------------
class Generator(nn.Module):
    """MLP generator: z -> joint vector (x, y)."""

    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Critic(nn.Module):
    """WGAN-GP critic (discriminator). Optionally returns feature for feature matching."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, return_feature: bool = False):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        feature = F.relu(self.fc3(x))
        score = self.fc4(feature)
        if return_feature:
            return score, feature
        return score


def get_mixed_noise(
    batch_size: int,
    latent_dim: int,
    device: torch.device,
    std: float = 2.0,
    ratio: float = 0.5,
) -> torch.Tensor:
    """
    Mixed Gaussian–uniform noise used in the study.
    z = std * (ratio * N(0,1) + (1-ratio) * U(-1,1))
    """
    gaussian = torch.randn(batch_size, latent_dim, device=device)
    uniform = 2.0 * torch.rand(batch_size, latent_dim, device=device) - 1.0
    z = ratio * gaussian + (1.0 - ratio) * uniform
    return z * std


def gradient_penalty(
    critic: nn.Module,
    real_samples: torch.Tensor,
    fake_samples: torch.Tensor,
    lambda_gp: float,
) -> torch.Tensor:
    """
    WGAN-GP gradient penalty.
    GP = lambda_gp * E[(||∇_x D(x̂)||_2 - 1)^2], x̂ = alpha*real + (1-alpha)*fake
    """
    device = real_samples.device
    batch_size = real_samples.size(0)

    alpha = torch.rand(batch_size, 1, device=device)
    alpha = alpha.expand_as(real_samples)

    interpolates = alpha * real_samples + (1.0 - alpha) * fake_samples
    interpolates = interpolates.requires_grad_(True)

    d_interpolates = critic(interpolates)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates, device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    gp = ((gradients.norm(2, dim=1) - 1.0) ** 2).mean() * float(lambda_gp)
    return gp

# -----------------------------
# Bayesian components (BNN)
# -----------------------------
class BayesianLinear(nn.Module):
    """
    Bayesian linear layer with factorized Gaussian weight uncertainty.

    Posterior parameterization (aligned with the study implementation):
        w = mu + exp(rho) * eps,   eps ~ N(0, 1)
        b = mu + exp(rho) * eps,   eps ~ N(0, 1)

    Regularization:
        A KL-like Gaussian regularization term inspired by the Bayesian formulation is adopted.
        For numerical stability and practical considerations, it is used as a regularizer
        rather than as part of a strictly derived ELBO objective. Therefore, the returned
        value may not behave as a strict KL divergence (e.g., it can take negative values
        depending on parameterization and scaling in the overall loss).
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)

        # Posterior parameters (aligned with the original code)
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-1.0, 1.0))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-3.0, -2.0))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-1.0, 1.0))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-3.0, -2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reparameterization trick (aligned with the original code)
        weight_eps = Normal(0.0, 1.0).sample(self.weight_mu.shape).to(self.weight_mu.device)
        bias_eps = Normal(0.0, 1.0).sample(self.bias_mu.shape).to(self.bias_mu.device)

        weight_sigma = torch.exp(self.weight_rho)
        bias_sigma = torch.exp(self.bias_rho)

        weight = self.weight_mu + weight_sigma * weight_eps
        bias = self.bias_mu + bias_sigma * bias_eps
        return F.linear(x, weight, bias)

    def kl_like(self) -> torch.Tensor:
        """
        KL divergence term used as a regularizer (KL-like).

        This expression follows the study code implementation, where rho is used directly in
        the regularization term. It is inspired by the KL between a factorized Gaussian posterior
        and a standard normal prior, but is not claimed as a strict ELBO-derived KL divergence.

        Note:
            The returned value can be negative depending on parameterization and scaling.
        """
        # aligned with your original kl_divergence() implementation
        weight_reg = 0.5 * torch.sum(
            1.0 + self.weight_rho - self.weight_mu.pow(2) - torch.exp(self.weight_rho)
        )
        bias_reg = 0.5 * torch.sum(
            1.0 + self.bias_rho - self.bias_mu.pow(2) - torch.exp(self.bias_rho)
        )
        return weight_reg + bias_reg


class BayesianNeuralNetwork(nn.Module):
    """
    Simple MLP-style Bayesian neural network (aligned with the study code):
    - BayesianLinear layers with exp(rho) parameterization
    - leaky ReLU activations
    """

    def __init__(
        self,
        input_features: int,
        hidden_units_1: int,
        hidden_units_2: int,
        hidden_units_3: int,
        output_features: int,
        negative_slope: float = 0.05,
    ):
        super().__init__()
        self.negative_slope = float(negative_slope)

        self.linear1 = BayesianLinear(input_features, hidden_units_1)
        self.linear2 = BayesianLinear(hidden_units_1, hidden_units_2)
        self.linear3 = BayesianLinear(hidden_units_2, hidden_units_3)
        self.linear4 = BayesianLinear(hidden_units_3, output_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.linear1(x), negative_slope=self.negative_slope)
        x = F.leaky_relu(self.linear2(x), negative_slope=self.negative_slope)
        x = F.leaky_relu(self.linear3(x), negative_slope=self.negative_slope)
        x = self.linear4(x)
        return x

    def kl_like(self) -> torch.Tensor:
        return (
            self.linear1.kl_like()
            + self.linear2.kl_like()
            + self.linear3.kl_like()
            + self.linear4.kl_like()
        )



