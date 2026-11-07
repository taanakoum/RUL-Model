"""Two inference modes for any model's RUL predictions (PROJECT_PLAN.md D23).

* ``current``: each prediction as the model makes it. The model already reads the past
  *signals* of the tool (its input window and, for B6 and RUL-Fuse, the causal memory),
  but never its own past *predictions*.
* ``context``: the model's past predictions of the same tool guide the new one. The
  running estimate is aged by the operating time that has passed (the remaining life falls
  by one second per second of cutting) and blended with the new prediction:

      aged_t = x_{t-1} − Δτ_t        (near the cap: never below r_t, see below)
      x_t    = aged_t + α_t (r_t − aged_t),   α_t = 1 − exp(−Δτ_t / T)

  where r_t is the model's prediction and T the time constant. A past prediction made Δ
  seconds of cutting ago keeps the weight exp(−Δ/T). The output is x_t clipped to [0, H].

Properties (tests/test_inference.py):

* **Causal.** Output t uses predictions 0..t of the same tool only: it is what a deployed
  model would have produced at that moment. A tool's context starts at its first point
  (a new tool resets it).
* **It can recover.** Every new prediction enters with weight α_t, so after an early
  prediction that was too low the output rises back towards the model's later predictions:
  the error left by a past mistake falls as exp(−elapsed/T), to 5 % after 3T = 360 s
  whatever its size. (A running minimum, which never rises, can only wait for the true RUL
  to fall to it: an error of e seconds lasts e seconds of cutting. It is not used.)
* **Exact on a right trajectory.** Below the cap the true RUL falls by exactly Δτ, so a
  model that predicts it exactly is left unchanged by the ageing and the blend.
* **The cap.** The target is min(RUL, H): above H it does not fall, and a prediction near H
  only says "at least about H". While the new prediction is within ``cap_margin_s`` of H,
  ageing therefore never takes the estimate below that prediction: aged_t =
  max(x_{t-1} − Δτ_t, min(x_{t-1}, r_t)). Without this, an estimate sitting at the cap would
  drift down by about T; pausing the ageing instead would make it lag by up to T when the
  RUL starts falling. With it, an exact trajectory is reproduced exactly, cap included.

Constants (fixed before any B6 or RUL-Fuse result, so comparing the two modes on the
validation tools is as fair as on the test tools): T = 120 s and the cap margin H/15. On
the real tools' time axes with synthetic errors, T = 120 s cut the window MAE of jitter
(46.6 → 8.5 s) and of pass-to-pass errors (72.6 → 45.3 s), could not remove a constant
per-tool bias (111.8 → 113.8 s), and on a mix of all four gave 142.8 → 133.5 s; T = 60 or
240 s did better on some errors and worse on others (PROJECT_LOG.md, 2026-11-07).
"""

from __future__ import annotations

import numpy as np

INFERENCE_MODES = ("current", "context")
CONTEXT_TIME_S = 120.0  # T: a past prediction's weight halves every T·ln 2 ≈ 83 s of cutting
CAP_MARGIN_FRACTION = 1.0 / 15.0  # a prediction within H/15 of H bounds the RUL from below


def context_filter(
    tau_s: np.ndarray,
    prediction_s: np.ndarray,
    horizon_s: float,
    time_constant_s: float = CONTEXT_TIME_S,
    cap_margin_s: float | None = None,
) -> np.ndarray:
    """The ``context`` mode of one tool's predictions, given in time order (``tau_s``
    non-decreasing, in seconds of operation)."""
    tau = np.asarray(tau_s, dtype=np.float64)
    raw = np.clip(np.asarray(prediction_s, dtype=np.float64), 0.0, horizon_s)
    if tau.shape != raw.shape:
        raise ValueError("tau_s and prediction_s must have the same length")
    if len(raw) == 0:
        return raw.copy()
    if np.any(np.diff(tau) < 0):
        raise ValueError("predictions must be in time order")
    if time_constant_s <= 0:
        raise ValueError("time_constant_s must be positive")
    margin = horizon_s * CAP_MARGIN_FRACTION if cap_margin_s is None else cap_margin_s
    out = np.empty_like(raw)
    x = raw[0]
    out[0] = x
    for t in range(1, len(raw)):
        dt = tau[t] - tau[t - 1]
        aged = x - dt
        if raw[t] >= horizon_s - margin:  # near the cap: not below the new prediction
            aged = max(aged, min(x, raw[t]))
        alpha = -np.expm1(-dt / time_constant_s)
        x = aged + alpha * (raw[t] - aged)
        out[t] = min(max(x, 0.0), horizon_s)
    return out


def apply_mode(
    mode: str,
    tau_s: np.ndarray,
    prediction_s: np.ndarray,
    horizon_s: float,
    time_constant_s: float = CONTEXT_TIME_S,
) -> np.ndarray:
    """One tool's predictions (time order) in the given inference mode."""
    if mode == "current":
        return np.clip(np.asarray(prediction_s, dtype=np.float64), 0.0, horizon_s)
    if mode == "context":
        return context_filter(tau_s, prediction_s, horizon_s, time_constant_s)
    raise ValueError(f"unknown inference mode {mode!r}; use one of {INFERENCE_MODES}")
