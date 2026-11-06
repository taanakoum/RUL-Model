"""Temporal losses on pairs of predictions of one unit (PROJECT_PLAN.md D22, §6.7).

Remaining life can only fall as operating time passes. For a pair of prediction points of
the same unit, (earlier, later), with Δτ seconds of operation between them:

* ``monotonic_hinge``: relu(ŷ_later − ŷ_earlier). A predicted RUL that rises over time is
  never right; a flat one is right in the capped region (RUL > H, where y stays at H).
* ``slope_consistency``: ((ŷ_earlier − ŷ_later) − Δτ)², only where the earlier target is
  below the cap. There the true RUL falls by exactly Δτ, so the prediction should too.

Everything is in the same units (here y/H and Δτ/H). Means are over the valid pairs.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _mean(values: Tensor, mask: Tensor | None) -> Tensor:
    if mask is None:
        return values.mean() if values.numel() else values.new_zeros(())
    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def monotonic_hinge(earlier: Tensor, later: Tensor, mask: Tensor | None = None) -> Tensor:
    return _mean(torch.relu(later - earlier), mask)


def slope_consistency(
    earlier: Tensor, later: Tensor, elapsed: Tensor, mask: Tensor | None = None
) -> Tensor:
    return _mean(((earlier - later) - elapsed) ** 2, mask)
