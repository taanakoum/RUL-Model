"""CUDA is a separate acceptance gate: CPU determinism cannot certify it."""

from __future__ import annotations

import pyarrow.parquet as pq
import pytest
import torch

from rulfuse.train import fit
from tests.factories import MICRO

pytestmark = [
    pytest.mark.integration,
    pytest.mark.gpu,
    pytest.mark.slow,
    pytest.mark.requirements("P14", "D26"),
]


@pytest.mark.parametrize("fusion", ["F0", "F1"])
def test_cuda_resume_reproduces_selected_weights_and_predictions(micro_bundle, tmp_path, fusion):
    b = micro_bundle
    config = MICRO.with_overrides({"fusion": fusion, "dropout": 0.15, "regularize": True})
    args = (b.store, b.signal, b.image, b.folds[0], 0, config)
    kw = {"device": "cuda", "log": lambda *_: None}
    fit(*args, tmp_path / "full", **kw)
    fit(*args, tmp_path / "resume", stop_after_steps=5, **kw)
    checkpoint = torch.load(
        tmp_path / "resume/checkpoint.pt", weights_only=False, map_location="cpu"
    )
    assert checkpoint["torch_rng"].device.type == "cpu"
    assert checkpoint["cuda_rng"] and all(t.device.type == "cpu" for t in checkpoint["cuda_rng"])
    fit(*args, tmp_path / "resume", **kw)
    full = torch.load(tmp_path / "full/model.pt", weights_only=True, map_location="cpu")
    resumed = torch.load(tmp_path / "resume/model.pt", weights_only=True, map_location="cpu")
    assert full.keys() == resumed.keys()
    assert all(torch.equal(full[k], resumed[k]) for k in full)
    assert pq.read_table(tmp_path / "full/predictions.parquet").equals(
        pq.read_table(tmp_path / "resume/predictions.parquet")
    )
