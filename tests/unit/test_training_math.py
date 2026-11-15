"""P12/P14/P17: exact loss masks, detached references, and equal trajectory weight."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rulfuse.models.heads import student_t_nll
from rulfuse.train import _ema_update, loss_terms, pair_terms
from tests.factories import MICRO

pytestmark = [pytest.mark.unit, pytest.mark.requirements("P12", "P14", "P17", "D22")]


def make_loss_inputs():
    mu = torch.tensor([[.4, .6, .8], [.5, .4, .2]], requires_grad=True)
    out = {"mu": mu, "sigma": torch.full_like(mu, .2, requires_grad=True),
           "nu": torch.full_like(mu, 5., requires_grad=True),
           "hi": torch.zeros_like(mu, requires_grad=True)}
    batch = {"hi_valid": torch.zeros_like(mu, dtype=torch.bool), "hi": torch.zeros_like(mu),
             "y": torch.tensor([[2., 4., 0.], [3., 2., 1.]]),
             "weight": torch.tensor([[.5, .5, 0.], [1/3, 1/3, 1/3]]),
             "point": torch.tensor([[True, True, False], [True, True, True]]),
             "tau": torch.tensor([[1., 2., 3.], [1., 2., 3.]])}
    return out, batch


def test_mse_equals_average_of_per_unit_weighted_mse():
    out, batch = make_loss_inputs()
    cfg = MICRO.with_overrides({"mono_weight": 0., "cons_weight": 0., "hi_weight": 0.})
    loss, _ = loss_terms(out, batch, 10., cfg, "B", 0., 2)
    unit_a = ((.4 - .2)**2 + (.6 - .4)**2) / 2
    unit_b = ((.5 - .3)**2 + (.4 - .2)**2 + (.2 - .1)**2) / 3
    assert loss.item() == pytest.approx((unit_a + unit_b) / 2)


def test_post_failure_predictions_cannot_contribute_to_rul_losses():
    out, batch = make_loss_inputs()
    changed = {key: value.clone() for key, value in out.items()}
    changed["mu"][0, 2] = 1000.
    for phase in ["A", "B"]:
        a, _ = loss_terms(out, batch, 10., MICRO, phase, 1., 2)
        b, _ = loss_terms(changed, batch, 10., MICRO, phase, 1., 2)
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_phase_a_uses_only_health_targets():
    out, batch = make_loss_inputs()
    batch["hi_valid"][0, 0] = True
    batch["hi"][0, 0] = 2.
    loss, parts = loss_terms(out, batch, 10., MICRO, "A", 10., 2)
    assert loss.item() == pytest.approx(1.5)
    loss.backward()
    assert out["mu"].grad is None and out["sigma"].grad is None
    assert set(parts) == {"hi"}


def test_nll_detaches_mean_but_trains_both_spread_parameters():
    mu = torch.tensor([.25, .8], requires_grad=True)
    sigma = torch.tensor([.1, .2], requires_grad=True)
    nu = torch.tensor([4., 6.], requires_grad=True)
    student_t_nll(torch.tensor([.5, .2]), mu.detach(), sigma, nu).mean().backward()
    assert mu.grad is None
    assert sigma.grad is not None and torch.isfinite(sigma.grad).all() and sigma.grad.abs().sum() > 0
    assert nu.grad is not None and torch.isfinite(nu.grad).all() and nu.grad.abs().sum() > 0


def test_temporal_pairs_respect_cap_mask_and_maximum_physical_lag():
    mu = torch.tensor([[.9, .8, .7]], requires_grad=True)
    batch = {"point": torch.ones(1, 3, dtype=torch.bool),
             "tau": torch.tensor([[0., 1., 20.]]), "y": torch.tensor([[10., 8., 0.]])}
    # First pair starts at cap: slope ignored. Other pairs exceed 0.1333*H.
    hinge, slope = pair_terms(mu, batch, 10., MICRO, 2)
    assert hinge.item() == 0. and slope.item() == 0.


@pytest.mark.parametrize("decay", [0., .5, .99])
def test_ema_parameter_update_has_exact_expected_value(decay):
    model, ema = torch.nn.Linear(2, 1), torch.nn.Linear(2, 1)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(4.)
        for parameter in ema.parameters():
            parameter.fill_(2.)
    _ema_update(ema, model, decay)
    for parameter in ema.parameters():
        np.testing.assert_allclose(parameter.detach().numpy(), 2 * decay + 4 * (1 - decay))
