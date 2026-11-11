"""Output heads and the Student-t likelihood (PROJECT_PLAN.md §6.6, §6.7; part P12).

The model works in units of y/H. The Student-t head gives:

* μ = sigmoid(·), in (0, 1), so μ·H ∈ (0, H);
* σ = softplus(·) + 0.005;
* ν = 3 + softplus(·), which keeps the variance finite.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

SIGMA_FLOOR = 0.005
NU_MIN = 3.0


class StudentTHead(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, 3)
        with torch.no_grad():
            self.linear.bias.copy_(torch.tensor([0.5, -1.0, 0.0]))

    def forward(self, h: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        out = self.linear(h)
        mu = torch.sigmoid(out[..., 0])
        sigma = F.softplus(out[..., 1]) + SIGMA_FLOOR
        nu = NU_MIN + F.softplus(out[..., 2])
        return mu, sigma, nu


def student_t_nll(y: Tensor, mu: Tensor, sigma: Tensor, nu: Tensor) -> Tensor:
    """Elementwise negative log-likelihood of a location-scale Student-t."""
    z = (y - mu) / sigma
    return (
        torch.lgamma(nu / 2)
        - torch.lgamma((nu + 1) / 2)
        + 0.5 * torch.log(nu * math.pi)
        + torch.log(sigma)
        + (nu + 1) / 2 * torch.log1p(z**2 / nu)
    )


def student_t_variance(sigma: Tensor, nu: Tensor) -> Tensor:
    return sigma**2 * nu / (nu - 2)
