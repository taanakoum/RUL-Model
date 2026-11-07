"""Temporal losses on pairs of predictions of one unit (PROJECT_PLAN.md D22, §6.7).

Remaining life can only fall as operating time passes. For a pair of prediction points of
the same unit, (earlier, later), with Δτ seconds of operation between them:

* ``monotonic_hinge``: relu(ŷ_later − ŷ_earlier). A predicted RUL that rises over time is
  never right; a flat one is right in the capped region (RUL > H, where y stays at H).
* ``slope_consistency``: ((ŷ_earlier − ŷ_later) − Δτ)², only where the earlier target is
  below the cap. There the true RUL falls by exactly Δτ, so the prediction should too.

**The later prediction is the reference: no gradient flows into it from these terms**
(it is detached). The later prediction has seen more of the unit, and the terms only move
the earlier one towards it. So an earlier prediction the model cannot correct (one that is
too low, say) can never drag later predictions down with it; the later ones are trained by
their own targets alone. Without the detach, with weight 1 on the hinge, the later
prediction of such a pair settled exactly at the stuck earlier value (PROJECT_LOG.md,
2026-11-07).

The weights (``mono_weight``, ``slope_weight`` in the trainers) must stay small with the
detach: the hinge only ever lifts earlier predictions, so prediction noise the model cannot
remove shifts the level up by about weight × (share of rising pairs) × H/2. At the weights
used (1/15 and 0.05) that is about 10-15 s; at weight 1 it would be 160-215 s.

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
    """mean relu(ŷ_later − ŷ_earlier); the gradient reaches ``earlier`` only."""
    return _mean(torch.relu(later.detach() - earlier), mask)


def slope_consistency(
    earlier: Tensor, later: Tensor, elapsed: Tensor, mask: Tensor | None = None
) -> Tensor:
    """mean ((ŷ_earlier − ŷ_later) − Δτ)²; the gradient reaches ``earlier`` only."""
    return _mean(((earlier - later.detach()) - elapsed) ** 2, mask)
