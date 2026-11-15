"""B6 seed membership must agree by identity, not merely by a count per point."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("omnifuser", reason="B6 needs the omnifuser extra")
pytest.importorskip("torchvision", reason="B6 needs torchvision")

from rulfuse.omnifuser_rul.aggregate import ensemble  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.adversarial, pytest.mark.regression,
              pytest.mark.requirements("P15", "P21", "D20")]


def rows():
    return {"role": np.array(["validation"] * 4), "fold": np.ones(4, int),
            "row": np.array([0, 1, 0, 1]), "seed": np.array([0, 0, 1, 1]),
            "unit": np.array(["A"] * 4), "on_grid": np.ones(4, bool),
            "prediction_s": np.array([10., 20., 12., 22.])}


@pytest.mark.parametrize("corruption", ["duplicate-seed", "disjoint-seeds", "wrong-unit", "grid"])
def test_b6_rejects_corrupt_members_even_if_every_point_has_the_same_count(corruption):
    data = rows()
    if corruption == "duplicate-seed":
        data["seed"][:] = 0
    elif corruption == "disjoint-seeds":
        data["row"] = np.array([0, 0, 1, 1])
    elif corruption == "wrong-unit":
        data["unit"][2:] = "B"
    else:
        data["on_grid"][2:] = False
    with pytest.raises(ValueError):
        ensemble(data, "validation")


def test_b6_ensemble_is_invariant_to_storage_row_order():
    data = rows()
    order = [3, 0, 2, 1]
    a = ensemble(data, "validation")
    b = ensemble({k: v[order] for k, v in data.items()}, "validation")
    for field in ("row", "unit", "on_grid", "prediction_s", "seeds"):
        np.testing.assert_array_equal(a[field], b[field])
