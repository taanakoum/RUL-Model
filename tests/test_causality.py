"""Models F1 and F0: causality, masks and the product of experts (PROJECT_PLAN.md part P11),
and the heads (P12)."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from scipy import stats

from rulfuse.models.fusion import build_model
from rulfuse.models.heads import StudentTHead, student_t_nll
from rulfuse.sequences import FoldTensors, collate, make_sequence
from rulfuse.tokens import build_fold_data
from rulfuse.train import loss_terms, predict
from tests.toy_engine import TINY, build_toy

FIELDS = (
    "seg",
    "tau",
    "elapsed",
    "entry",
    "present",
    "point",
    "rul",
    "y",
    "window",
    "weight",
    "hi",
    "oracle",
)


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    t = build_toy(tmp_path_factory.mktemp("toy_causal"))
    fold = t.folds[0]
    out = {}
    for fusion in ("F1", "F0"):
        cfg = TINY.with_overrides({"fusion": fusion})
        data = build_fold_data(t.store, t.signal, t.image, fold, cfg)
        torch.manual_seed(0)
        model = build_model(cfg, data, t.store.settings.horizon_s).eval()
        out[fusion] = (
            cfg,
            data,
            model,
            FoldTensors(data, [fold.validation], "cpu"),
            fold.validation,
        )
    return out


def _truncate(seq, cut):
    short = copy.copy(seq)
    for name in FIELDS:
        setattr(short, name, getattr(seq, name)[:cut])
    keep = seq.img_time < seq.tau[cut - 1]
    short.img, short.img_time, short.img_entry = (
        seq.img[keep],
        seq.img_time[keep],
        seq.img_entry[keep],
    )
    return short


@pytest.mark.parametrize("fusion", ["F1", "F0"])
def test_truncation_invariance(setup, fusion):
    cfg, data, model, tensors, unit = setup[fusion]
    full = make_sequence(data.units[unit])
    a = predict(model, full, tensors, cfg, len(data.channels))
    for cut in (5, len(full.seg) // 3, len(full.seg) - 3):
        b = predict(model, _truncate(full, cut), tensors, cfg, len(data.channels))
        assert np.allclose(a.mu_s[:cut], b.mu_s, atol=1e-4)
        assert np.allclose(a.sigma_s[:cut], b.sigma_s, atol=1e-4)


@pytest.mark.parametrize("fusion", ["F1", "F0"])
def test_no_image_leak_before_the_image_is_available(setup, fusion):
    cfg, data, model, tensors, unit = setup[fusion]
    seq = make_sequence(data.units[unit])
    j = len(seq.img) // 2
    when = seq.img_time[j]
    a = predict(model, seq, tensors, cfg, len(data.channels))
    changed = FoldTensors(data, [unit], "cpu")
    changed.images[seq.img[j]] += 5.0
    b = predict(model, seq, changed, cfg, len(data.channels))
    before = seq.tau <= when + 1e-6
    assert np.allclose(a.mu_s[before], b.mu_s[before], atol=1e-5)
    assert not np.allclose(a.mu_s[~before], b.mu_s[~before], atol=1e-5)


def test_f0_with_sensing_dropped_equals_the_usage_expert(setup):
    cfg, data, model, tensors, unit = setup["F0"]
    seq = make_sequence(data.units[unit])
    batch = collate([seq], tensors, cfg.with_overrides({"players": ()}), None, len(data.channels))
    with torch.no_grad():
        out = model(batch, tensors)
    ex = out["experts"]
    assert torch.allclose(out["alpha"][..., 0], torch.ones_like(out["alpha"][..., 0]))
    assert torch.allclose(out["mu"], ex["mu"][..., 0], atol=1e-6)
    var = ex["sigma"][..., 0] ** 2 * ex["nu"][..., 0] / (ex["nu"][..., 0] - 2)
    fused = out["sigma"] ** 2 * out["nu"] / (out["nu"] - 2)
    assert torch.allclose(var, fused, rtol=1e-4)


def test_missing_modality_masks(setup):
    cfg, data, model, tensors, unit = setup["F1"]
    seq = make_sequence(data.units[unit])
    no_img = cfg.with_overrides({"players": ("signal",)})
    batch = collate([seq], tensors, no_img, None, len(data.channels))
    assert not batch["img_valid"].any()
    no_sig = cfg.with_overrides({"players": ("surface",)})
    batch = collate([seq], tensors, no_sig, None, len(data.channels))
    assert not batch["present"].any()


def test_student_t_head_ranges_and_nll():
    head = StudentTHead(8)
    h = torch.randn(50, 8) * 10
    mu, sigma, nu = head(h)
    assert ((mu > 0) & (mu < 1)).all() and (sigma >= 0.005).all() and (nu >= 3).all()
    y = torch.randn(50)
    ours = student_t_nll(y, mu, sigma, nu).detach().numpy()
    ref = -stats.t.logpdf(
        y.numpy(), nu.detach().numpy(), mu.detach().numpy(), sigma.detach().numpy()
    )
    assert np.allclose(ours, ref, atol=1e-4)


def test_nll_never_moves_the_mean(setup):
    cfg, data, model, tensors, unit = setup["F1"]
    seq = make_sequence(data.units[unit])
    batch = collate([seq], tensors, cfg, None, len(data.channels))
    model.train()
    out = model(batch, tensors)
    mu = out["mu"]
    mu.retain_grad()
    only_nll = cfg.with_overrides({"mono_weight": 0.0, "cons_weight": 0.0, "hi_weight": 0.0})
    loss, parts = loss_terms(out, batch, data.units[unit].horizon_s, only_nll, "B", 1.0, 2)
    (loss - parts["mse"] * 0).backward(retain_graph=True)
    g_total = mu.grad.clone()
    model.zero_grad()
    mu.grad = None
    y = batch["y"] / data.units[unit].horizon_s
    w = batch["weight"] * batch["point"]
    mse = ((mu - y) ** 2 * w).sum() / w.sum()
    mse.backward()
    assert mu.grad is not None
    assert torch.allclose(g_total, mu.grad, atol=1e-7)
    model.eval()
