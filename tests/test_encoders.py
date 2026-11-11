"""P8 Encoders (PROJECT_PLAN.md §6.2): shapes, determinism, pooling and signatures.

The Mantis-8M and DINOv2 wrappers are tested with small randomly initialized networks, so
no weights are downloaded. Tests marked ``weights`` load the real checkpoints and run only
where they are cached (on Ruche, in the step-0 job, they prove offline loading).
"""

from __future__ import annotations

import numpy as np
import pytest

from rulfuse.features.encoders import (
    DINOV2_REPO,
    GRID,
    MANTIS_REPO,
    PixelStatsEncoder,
    ProjectionEncoder,
    hf_revision,
)


def test_projection_encoder_is_deterministic_and_scale_free():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, 512))
    a, b = ProjectionEncoder(), ProjectionEncoder()
    np.testing.assert_array_equal(a.encode(x), b.encode(x))
    np.testing.assert_allclose(a.encode(3 * x + 7), a.encode(x), atol=1e-5)
    assert a.encode(x).shape == (5, 16)
    assert ProjectionEncoder(seed=1).signature() != a.signature()


def test_pixel_stats_tokens():
    x = np.zeros((2, 3, 28, 56), np.float32)
    x[:, :, :14, :14] = 1.0  # the top-left grid cell only
    tokens = PixelStatsEncoder().encode(x)
    assert tokens.shape == (2, 1 + GRID[0] * GRID[1], 6)
    np.testing.assert_allclose(tokens[:, 1, :3], 1.0)  # top-left cell mean
    np.testing.assert_allclose(tokens[:, 2, :3], 0.0)


def test_mantis_wrapper_with_a_random_network():
    pytest.importorskip("mantis")
    from mantis.architecture import MantisV1

    from rulfuse.features.encoders import MantisEncoder

    encoder = MantisEncoder(device="cpu", batch_size=4, network=MantisV1(device="cpu"))
    x = np.random.default_rng(0).normal(size=(6, 512)).astype(np.float32)
    out = encoder.encode(x)
    assert out.shape == (6, 256) and out.dtype == np.float32
    np.testing.assert_allclose(encoder.encode(x), out, atol=1e-5)  # eval mode: deterministic
    assert encoder.signature()["repo"] == "random-init"


def test_dinov2_wrapper_pools_the_patch_grid():
    pytest.importorskip("transformers")
    from transformers import Dinov2Config, Dinov2Model

    from rulfuse.features.encoders import DinoV2Encoder

    config = Dinov2Config(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        mlp_ratio=2,
        image_size=56,
    )
    encoder = DinoV2Encoder(device="cpu", batch_size=2, model=Dinov2Model(config))
    pixels = np.random.default_rng(0).normal(size=(3, 3, 28, 56)).astype(np.float32)
    tokens = encoder.encode(pixels)
    assert tokens.shape == (3, 9, 16)
    import torch

    with torch.no_grad():
        hidden = encoder.model(pixel_values=torch.from_numpy(pixels)).last_hidden_state.numpy()
    np.testing.assert_allclose(tokens[:, 0], hidden[:, 0], atol=1e-5)  # the class token
    patches = hidden[:, 1:].reshape(3, 2, 4, 16)  # 28 × 56 px = 2 × 4 patches of 14 px
    np.testing.assert_allclose(tokens[:, 1:], patches.reshape(3, 8, 16), atol=1e-5)
    np.testing.assert_allclose(encoder.encode(pixels), tokens, atol=1e-5)


@pytest.mark.weights
def test_real_mantis_loads_offline():
    if hf_revision(MANTIS_REPO) is None:
        pytest.skip("Mantis-8M not cached")
    from rulfuse.features.encoders import MantisEncoder

    encoder = MantisEncoder(device="cpu", batch_size=8)
    assert encoder.signature()["revision"] is not None
    assert encoder.encode(np.zeros((2, 512), np.float32)).shape == (2, 256)


@pytest.mark.weights
def test_real_dinov2_loads_offline():
    if hf_revision(DINOV2_REPO) is None:
        pytest.skip("DINOv2-B not cached")
    from rulfuse.features.encoders import DinoV2Encoder

    encoder = DinoV2Encoder(device="cpu", batch_size=2)
    assert encoder.dim == 768
    assert encoder.encode(np.zeros((1, 3, 224, 224), np.float32)).shape == (1, 9, 768)


def test_a_weights_only_cache_still_gives_the_revision(tmp_path, monkeypatch):
    """Ruche, 2026-11-11: the prefetch caches only *.json and *.safetensors, and
    huggingface_hub raised IncompleteSnapshotError (README, logo missing) for an unfiltered
    local lookup, so J0 saw no revision. The lookup must work on such a cache."""
    from huggingface_hub import constants

    commit = "bc7d5ab40c02133386a28e2c127f35c17c86901d"
    folder = tmp_path / "models--org--model"
    (folder / "refs").mkdir(parents=True)
    (folder / "refs" / "main").write_text(commit)
    snapshot = folder / "snapshots" / commit
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert hf_revision("org/model") is None  # no weight file yet
    (snapshot / "model.safetensors").write_bytes(b"0")
    assert hf_revision("org/model") == commit
    assert hf_revision("org/absent") is None
