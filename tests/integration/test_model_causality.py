"""D9/D14/P10-P12: perturb every unavailable stream, not just the final photograph."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from rulfuse.models.fusion import SignalEncoder, attention_mask, build_model
from rulfuse.sequences import FoldTensors, collate, make_sequence
from rulfuse.tokens import build_fold_data
from tests.factories import MICRO

pytestmark = [pytest.mark.integration, pytest.mark.requirements("P10", "P11", "P12", "D9", "D14")]


def inputs(bundle, fusion, **overrides):
    config = MICRO.with_overrides({"fusion": fusion, **overrides})
    data = build_fold_data(bundle.store, bundle.signal, bundle.image, bundle.folds[0], config)
    name = bundle.folds[0].validation
    tensors = FoldTensors(data, [name], "cpu")
    sequence = make_sequence(data.units[name])
    batch = collate([sequence], tensors, config, None, len(data.channels))
    torch.manual_seed(207)
    model = build_model(config, data, bundle.store.settings.horizon_s).eval()
    return config, data, model, tensors, sequence, batch


@pytest.mark.regression
@pytest.mark.parametrize("fusion", ["F0", "F1"])
@pytest.mark.parametrize("mode", ["both", "relative", "absolute"])
def test_first_future_photo_cannot_change_any_earlier_output(micro_bundle, fusion, mode):
    _, _, model, tensors, sequence, batch = inputs(micro_bundle, fusion, token_mode=mode)
    before = sequence.tau <= sequence.img_time[0] + 1e-6
    assert before.any() and (~before).any()
    with torch.no_grad():
        original = model(batch, tensors)
        tensors.images[0] += 20.
        tensors.img_readout[0] += 20.
        changed = model(batch, tensors)
    for field in ("mu", "sigma", "nu", "hi"):
        torch.testing.assert_close(original[field][0, before], changed[field][0, before],
                                   rtol=0, atol=0, msg=f"{fusion}/{mode}/{field} reads future photo")


@pytest.mark.parametrize("fusion", ["F0", "F1"])
@pytest.mark.parametrize("memory", [False, True])
def test_future_signal_tokens_cannot_change_outputs_or_gradients(micro_bundle, fusion, memory):
    _, _, model, tensors, sequence, batch = inputs(micro_bundle, fusion, memory=memory)
    cut = len(sequence.seg) // 2
    with torch.no_grad():
        original = model(batch, tensors)
        tensors.signal[cut:] += 100.
        changed = model(batch, tensors)
    for field in ("mu", "sigma", "nu", "hi"):
        torch.testing.assert_close(original[field][:, :cut], changed[field][:, :cut], rtol=0, atol=0)
    tensors.signal.requires_grad_(True)
    model(batch, tensors)["mu"][0, cut - 1].backward()
    assert tensors.signal.grad is not None
    assert torch.count_nonzero(tensors.signal.grad[cut:]) == 0


@pytest.mark.regression
@pytest.mark.parametrize("fusion", ["F0", "F1"])
def test_gradient_cannot_reach_a_photo_before_it_exists(micro_bundle, fusion):
    _, _, model, tensors, _, batch = inputs(micro_bundle, fusion)
    tensors.images.requires_grad_(True)
    model(batch, tensors)["mu"][0, 0].backward()
    assert tensors.images.grad is not None
    assert torch.count_nonzero(tensors.images.grad) == 0, "future photos have a causal gradient path"


@pytest.mark.parametrize("fusion", ["F0", "F1"])
def test_usage_off_is_invariant_to_a_common_shift_of_operating_time(micro_bundle, fusion):
    _, _, model, tensors, _, batch = inputs(micro_bundle, fusion, usage=False)
    shifted = {k: v.clone() for k, v in batch.items()}
    shifted["tau"] += 200.
    shifted["img_time"] += 200.
    with torch.no_grad():
        a, b = model(batch, tensors), model(shifted, tensors)
    for field in ("mu", "sigma", "nu", "hi"):
        torch.testing.assert_close(a[field], b[field], rtol=1e-5, atol=1e-6)


def test_attention_reads_no_same_time_image_or_future_signal(micro_bundle):
    _, _, _, _, _, batch = inputs(micro_bundle, "F1")
    _, allowed = attention_mask(batch, 0, True)
    n = batch["tau"].shape[1]
    signal = allowed[:, :n, :n]
    image = allowed[:, :n, n:]
    assert not (signal & (batch["tau"][:, None, :] > batch["tau"][:, :, None] + 1e-6)).any()
    assert not (image & (batch["img_time"][:, None, :] >= batch["tau"][:, :, None] - 1e-6)).any()


@pytest.mark.parametrize("fusion", ["F0", "F1"])
@pytest.mark.parametrize("players", [(), ("signal",), ("edge",), ("signal", "edge", "texture")])
def test_all_modality_variants_produce_finite_physical_predictions(micro_bundle, fusion, players):
    _, _, model, tensors, _, batch = inputs(micro_bundle, fusion, players=players)
    with torch.no_grad():
        result = model(batch, tensors)
    for field in ("mu", "sigma", "nu", "hi"):
        assert torch.isfinite(result[field]).all(), f"{fusion}/{players}/{field} is nonfinite"
    assert ((result["mu"] >= 0) & (result["mu"] <= 1)).all()
    assert (result["sigma"] > 0).all() and (result["nu"] >= 3).all()
    if fusion == "F0":
        torch.testing.assert_close(result["alpha"].sum(-1), torch.ones_like(result["mu"]))
        assert (result["alpha"][~result["experts"]["present"]] == 0).all()


def test_channel_set_pooling_is_equivariant_to_channel_order():
    channels = [("force", "x", "raw"), ("force", "y", "raw"), ("sound", "none", "raw")]
    permutation = [2, 0, 1]
    torch.manual_seed(4)
    a = SignalEncoder(5, channels, MICRO).eval()
    b = SignalEncoder(5, [channels[i] for i in permutation], MICRO).eval()
    b.load_state_dict(copy.deepcopy(a.state_dict()))
    x = torch.randn(6, 3, 5)
    keep = torch.tensor([[True, False, True]])
    present = torch.ones(1, 6, dtype=torch.bool)
    with torch.no_grad():
        out_a = a.pool(a.base(x)[None], keep, present)
        out_b = b.pool(b.base(x[:, permutation])[None], keep[:, permutation], present)
    np.testing.assert_allclose(out_a, out_b, atol=1e-6)
