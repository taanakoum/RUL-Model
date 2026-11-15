"""Checks on the prediction rows a seed ensemble is built from (test review T04, T08).

A ``DONE`` file says a fit finished, not that its rows are the ones asked for. Before any
score or calibration uses them, every member must hold the expected tool, strictly increasing
finite times, finite Student-t parameters (sigma > 0, nu > 2), and, across members, the same
points, tool, truth and window. Anything else raises ValueError instead of producing a
plausible number.
"""

from __future__ import annotations

import numpy as np


def checked_members(parts: list[dict], unit: str | None = None) -> list[dict]:
    """The members, each in time order, after the checks above. ``unit`` is the tool every
    row must belong to (if the rows carry a ``unit`` column)."""
    if not parts:
        raise ValueError("no seed member to ensemble")
    out = []
    for i, p in enumerate(parts):
        order = np.argsort(np.asarray(p["tau_s"], float), kind="stable")
        q = {k: np.asarray(v)[order] for k, v in p.items()}
        tau = q["tau_s"].astype(float)
        if len(tau) == 0:
            raise ValueError(f"member {i}: no prediction rows")
        if not np.isfinite(tau).all() or np.any(np.diff(tau) <= 0):
            raise ValueError(f"member {i}: duplicate or nonfinite prediction times")
        if "unit" in q:
            units = set(q["unit"].astype(str).tolist())
            if len(units) != 1 or (unit is not None and units != {str(unit)}):
                raise ValueError(f"member {i}: rows of {sorted(units)}, expected {unit!r}")
        mu, sigma, nu = (q[k].astype(float) for k in ("mu_s", "sigma_s", "nu"))
        if not (np.isfinite(mu).all() and np.isfinite(sigma).all() and np.isfinite(nu).all()):
            raise ValueError(f"member {i}: nonfinite predictions")
        if np.any(sigma <= 0) or np.any(nu <= 2):
            raise ValueError(f"member {i}: sigma must be > 0 and nu > 2")
        window = q["window"].astype(bool)
        if not np.isfinite(q["rul_s"].astype(float)[window]).all():
            raise ValueError(f"member {i}: nonfinite targets on window points")
        out.append(q)
    first = out[0]
    for i, q in enumerate(out[1:], 1):
        for key in ("tau_s", "unit", "rul_s", "window"):
            if key not in first:
                continue
            a, b = first[key], q[key]
            same = (
                np.array_equal(a.astype(float), b.astype(float), equal_nan=True)
                if key == "rul_s"
                else np.array_equal(a, b)
            )
            if not same:
                raise ValueError(f"member {i} disagrees with member 0 on {key}")
    return out
