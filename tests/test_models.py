"""Shared model components (PROJECT_PLAN.md D21, D22): the causal memory and the temporal
losses, used by baseline B6 now and by RUL-Fuse in Stage 4."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rulfuse import metrics as M  # noqa: E402
from rulfuse.models.losses import monotonic_hinge, slope_consistency  # noqa: E402
from rulfuse.models.memory import CausalMemory  # noqa: E402


@pytest.mark.parametrize("cell", ["gru", "lstm"])
@pytest.mark.parametrize("block", [1, 4])
def test_memory_never_reads_a_later_step(cell, block):
    torch.manual_seed(0)
    memory = CausalMemory(5, 8, block=block, cell=cell).eval()
    x = torch.randn(2, 23, 5)
    with torch.no_grad():
        base = memory(x)
        for t in (0, 3, 7, 12, 22):
            changed = x.clone()
            changed[:, t + 1 :] += 10.0  # everything after step t
            np.testing.assert_allclose(memory(changed)[:, : t + 1], base[:, : t + 1], atol=1e-6)


def test_memory_reads_only_complete_blocks():
    torch.manual_seed(0)
    memory = CausalMemory(3, 4, block=5).eval()
    x = torch.randn(1, 12, 3)
    with torch.no_grad():
        states = memory(x)
        assert torch.all(states[:, :4] == 0)  # no block is complete before step 4
        assert torch.allclose(states[:, 4], states[:, 8])  # steps 4-8 read block 0
        assert not torch.allclose(states[:, 8], states[:, 9])  # step 9 completes block 1
        changed = x.clone()
        changed[:, 9, 0] += 5.0  # the step that completes block 1 changes its own state only
        new = memory(changed)
        assert torch.allclose(new[:, :9], states[:, :9]) and not torch.allclose(
            new[:, 9], states[:, 9]
        )


def test_memory_sees_a_rise_shared_by_every_input():
    torch.manual_seed(0)
    memory = CausalMemory(3, 4, block=2).eval()
    x = torch.randn(1, 6, 3)
    with torch.no_grad():  # a common shift (all descriptors rising with wear) must register
        assert not torch.allclose(memory(x + 1.0)[:, -1], memory(x)[:, -1])


def test_memory_block_means_ignore_padding():
    memory = CausalMemory(2, 3, block=4)
    x = torch.arange(16.0).reshape(1, 8, 2)
    padded = torch.cat([x, torch.full((1, 4, 2), 99.0)], dim=1)
    blocks = memory.blocks(padded, torch.tensor([10]))
    np.testing.assert_allclose(blocks[0, :2], memory.blocks(x)[0], atol=1e-6)
    assert torch.allclose(blocks[0, 2], padded[0, 8:10].mean(dim=0))  # 2 valid steps of 4


def test_temporal_losses():
    earlier = torch.tensor([0.5, 0.5, 0.5], requires_grad=True)
    later = torch.tensor([0.4, 0.5, 0.6], requires_grad=True)
    hinge = monotonic_hinge(earlier, later)
    assert float(hinge.detach()) == pytest.approx(0.1 / 3)  # only the rising pair counts
    hinge.backward()
    assert later.grad is None  # the later prediction is the reference: never moved
    assert earlier.grad is not None and earlier.grad[2] < 0  # the earlier one is lifted
    assert earlier.grad[0] == 0 and earlier.grad[1] == 0  # falling and flat pairs: no push
    elapsed = torch.tensor([0.1, 0.0, 0.2])
    slope = slope_consistency(earlier, later, elapsed)
    earlier.grad = None
    slope.backward()
    assert later.grad is None and earlier.grad is not None
    assert earlier.grad[2] < 0  # 0.5 → 0.6 should have fallen by 0.2: lift the earlier one
    assert float(slope_consistency(earlier.detach(), later.detach(), elapsed)) == pytest.approx(
        (0 + 0 + 0.3**2) / 3
    )
    mask = torch.tensor([True, True, False])
    masked = slope_consistency(earlier.detach(), later.detach(), elapsed, mask)
    assert float(masked) == pytest.approx(0.0, abs=1e-12)  # both kept pairs fall by Δτ
    none = monotonic_hinge(earlier.detach(), later.detach(), torch.zeros(3, dtype=torch.bool))
    assert float(none) == 0.0


def test_temporal_stability_metric():
    tau = np.array([0.0, 10.0, 20.0, 30.0])
    perfect = np.array([100.0, 90.0, 80.0, 70.0])
    stats = M.temporal_stability(perfect, tau)
    assert stats == {"increase_rate": 0.0, "mean_increase_s": 0.0, "slope_error_s": 0.0}
    rising = np.array([100.0, 105.0, 80.0, 70.0])
    stats = M.temporal_stability(rising, tau)
    assert stats["increase_rate"] == pytest.approx(1 / 3)
    assert stats["mean_increase_s"] == pytest.approx(5 / 3)
    assert stats["slope_error_s"] == pytest.approx((15 + 15 + 0) / 3)
