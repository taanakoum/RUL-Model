"""Degradation extrapolation: the F5 machinery behind B5 and R1 (PROJECT_PLAN.md §8, §6.5;
part P16, §15.1).

Generic for any dataset with a health indicator (HI) that rises with operating time.

* Each training unit's smoothed HI is fitted with HI(τ) = α + β·(τ/s)^γ, γ >= 1, β > 0
  (:func:`fit_power_law`; ``s`` is a time scale that keeps the parameters well conditioned).
* :class:`DegradationPrior` is the Gaussian over (α, log β, log γ) of those fits, with the
  failure threshold θ (the median HI at failure over the training units) and the noise sd
  (the pooled residual sd).
* :func:`predict_rul` draws from the prior, importance-weights each draw by the Gaussian
  likelihood of the unit's HI history, and turns every draw into a RUL: its first crossing of
  θ minus the current time, clipped to [0, H] (crossed already: 0; never crossing: H). The
  forecast is the weighted mean and the weighted 5 % / 95 % quantiles. Without a history
  the prior alone speaks.

Times are the unit's operating time τ in seconds; nothing here knows what the HI measures.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import NamedTuple

import numpy as np
from scipy.optimize import least_squares

DRAWS = 5000
RIDGE = 1e-6
LOG_GAMMA_MAX = float(np.log(8.0))
LOG_BETA_RANGE = (-20.0, 20.0)
NOISE_FLOOR = 1e-3  # of the pooled HI range
LOG_TIME_CAP = 50.0
LEVELS = (0.05, 0.95)


def power_law(tau: np.ndarray, params: np.ndarray, time_scale: float = 1.0) -> np.ndarray:
    """HI(τ) = α + β·(τ/s)^γ for ``params`` = (α, log β, log γ)."""
    alpha, log_beta, log_gamma = params
    t = np.maximum(np.asarray(tau, dtype=np.float64), 0.0) / time_scale
    return alpha + np.exp(log_beta) * np.power(t, np.exp(log_gamma))


def fit_power_law(
    tau: np.ndarray, hi: np.ndarray, time_scale: float = 1.0
) -> tuple[np.ndarray, float]:
    """Least-squares fit of HI = α + β·(τ/s)^γ with γ >= 1 and β > 0 (bounded
    ``scipy.optimize.least_squares``, started from several γ). Returns ((α, log β, log γ),
    residual sd), the sd with the 3 fitted parameters taken off the degrees of freedom."""
    tau, hi = np.asarray(tau, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    ok = np.isfinite(tau) & np.isfinite(hi)
    tau, hi = tau[ok], hi[ok]
    if len(tau) < 3:
        raise ValueError("fit_power_law needs at least 3 finite points")
    t = np.maximum(tau, 0.0) / time_scale
    span = max(float(np.ptp(hi)), 1e-9)
    lower = np.array([hi.min() - span, LOG_BETA_RANGE[0], 0.0])
    upper = np.array([hi.max() + span, LOG_BETA_RANGE[1], LOG_GAMMA_MAX])

    def residual(p: np.ndarray) -> np.ndarray:
        return p[0] + np.exp(p[1]) * np.power(t, np.exp(p[2])) - hi

    best = None
    top = max(float(t.max()), 1e-9)
    for gamma in (1.0, 2.0, 4.0):
        scale = max(float(hi.max() - hi.min()), 1e-9) / top**gamma
        start = np.array([hi.min(), np.log(scale), np.log(gamma)])
        start = np.clip(start, lower + 1e-9, upper - 1e-9)
        fit = least_squares(residual, start, bounds=(lower, upper), x_scale=[span, 1.0, 1.0])
        if best is None or fit.cost < best.cost:
            best = fit
    assert best is not None
    dof = max(len(tau) - 3, 1)
    return best.x, float(np.sqrt(2.0 * best.cost / dof))


@dataclass(frozen=True)
class DegradationPrior:
    """Gaussian prior over (α, log β, log γ) with the failure threshold and the noise sd."""

    mean: np.ndarray
    cov: np.ndarray
    threshold: float
    noise_sd: float
    time_scale: float
    n_units: int

    @classmethod
    def fit(
        cls,
        units: Sequence[tuple[np.ndarray, np.ndarray]],
        failure_hi: Sequence[float],
        time_scale: float | None = None,
    ) -> DegradationPrior:
        """Fit the prior on training units: ``units`` = [(τ, smoothed HI)], ``failure_hi`` =
        the HI at each unit's failure. ``time_scale`` defaults to the largest τ seen."""
        if len(units) < 2:
            raise ValueError("a prior needs at least 2 units")
        scale = (
            float(time_scale)
            if time_scale is not None
            else max(float(np.max(np.asarray(t, dtype=np.float64))) for t, _ in units)
        )
        triples, sse, dof, spans = [], 0.0, 0, []
        for tau, hi in units:
            params, _ = fit_power_law(tau, hi, scale)
            ok = np.isfinite(tau) & np.isfinite(hi)
            r = power_law(np.asarray(tau)[ok], params, scale) - np.asarray(hi)[ok]
            triples.append(params)
            sse += float(np.sum(r**2))
            dof += max(int(ok.sum()) - 3, 1)
            spans.append(float(np.ptp(np.asarray(hi)[ok])))
        stack = np.asarray(triples)
        cov = np.cov(stack, rowvar=False) + RIDGE * np.eye(3)
        noise = max(float(np.sqrt(sse / dof)), NOISE_FLOOR * float(np.mean(spans)), 1e-12)
        return cls(
            stack.mean(axis=0),
            cov,
            float(np.median(np.asarray(failure_hi, dtype=np.float64))),
            noise,
            scale,
            len(units),
        )

    def with_noise(self, noise_sd: float) -> DegradationPrior:
        return replace(self, noise_sd=float(noise_sd))


class Forecast(NamedTuple):
    mean: float
    lower: float
    upper: float
    ess: float  # effective sample size of the importance weights


class _Draws(NamedTuple):
    params: np.ndarray  # (draws, 3), log γ clipped to >= 0
    crossing: np.ndarray  # (draws,) first time HI >= θ (τ units; 0: crossed at the start)


def _draw(prior: DegradationPrior, draws: int, seed: int) -> _Draws:
    rng = np.random.default_rng(seed)
    params = rng.multivariate_normal(prior.mean, prior.cov, size=draws, method="eigh")
    params[:, 2] = np.clip(params[:, 2], 0.0, LOG_GAMMA_MAX)  # γ >= 1
    gap = prior.threshold - params[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        log_t = np.log(np.maximum(gap, 1e-300)) - params[:, 1]
        log_t = np.minimum(log_t / np.exp(params[:, 2]), LOG_TIME_CAP)
    crossing = np.where(gap > 0, prior.time_scale * np.exp(log_t), 0.0)
    return _Draws(params, crossing)


def _cumulative_loglik(
    prior: DegradationPrior, draws: _Draws, tau_hist: np.ndarray, hi_hist: np.ndarray
) -> np.ndarray:
    """Row n: log-likelihood of the first n history points under every draw (row 0: zeros)."""
    out = np.zeros((len(tau_hist) + 1, len(draws.params)))
    if len(tau_hist):
        t = np.maximum(tau_hist, 0.0)[:, None] / prior.time_scale
        model = draws.params[None, :, 0] + np.exp(draws.params[None, :, 1]) * np.power(
            t, np.exp(draws.params[None, :, 2])
        )
        out[1:] = np.cumsum(-0.5 * ((model - hi_hist[:, None]) / prior.noise_sd) ** 2, axis=0)
    return out


def _forecast(
    rul: np.ndarray, order: np.ndarray, log_w: np.ndarray
) -> Forecast:  # ``order`` sorts the draws by crossing time (so by RUL)
    w = np.exp(log_w - log_w.max())
    w /= w.sum()
    rul_sorted, w_sorted = rul[order], w[order]
    cum = np.cumsum(w_sorted)
    lo, hi = (
        float(rul_sorted[min(int(np.searchsorted(cum, q)), len(cum) - 1)]) for q in LEVELS
    )
    return Forecast(float(w @ rul), lo, hi, float(1.0 / np.sum(w**2)))


def predict_rul_many(
    prior: DegradationPrior,
    tau_hist: np.ndarray,
    hi_hist: np.ndarray,
    tau_now: np.ndarray,
    n_visible: np.ndarray,
    horizon_s: float,
    draws: int = DRAWS,
    seed: int = 0,
) -> list[Forecast]:
    """:func:`predict_rul` for many points of one unit. ``tau_hist`` is the unit's HI history
    in time order; point i sees its first ``n_visible[i]`` entries. The same prior draws serve
    every point."""
    tau_hist = np.asarray(tau_hist, dtype=np.float64)
    hi_hist = np.asarray(hi_hist, dtype=np.float64)
    if len(tau_hist) != len(hi_hist) or np.any(np.diff(tau_hist) < 0):
        raise ValueError("the history must be equal-length arrays in time order")
    sample = _draw(prior, draws, seed)
    log_lik = _cumulative_loglik(prior, sample, tau_hist, hi_hist)
    order = np.argsort(sample.crossing, kind="stable")
    out = []
    for now, n in zip(np.asarray(tau_now, dtype=np.float64), np.asarray(n_visible), strict=True):
        rul = np.clip(sample.crossing - now, 0.0, horizon_s)
        out.append(_forecast(rul, order, log_lik[int(n)]))
    return out


def predict_rul(
    prior: DegradationPrior,
    tau_hist: np.ndarray,
    hi_hist: np.ndarray,
    tau_now: float,
    horizon_s: float,
    draws: int = DRAWS,
    seed: int = 0,
) -> Forecast:
    """RUL forecast at ``tau_now`` from the HI history observed so far (all of it is used;
    the caller decides what is visible). Returns the weighted mean, the weighted 5 % and
    95 % quantiles and the effective sample size; with no history, the prior alone."""
    tau_hist = np.atleast_1d(np.asarray(tau_hist, dtype=np.float64))
    hi_hist = np.atleast_1d(np.asarray(hi_hist, dtype=np.float64))
    order = np.argsort(tau_hist, kind="stable")
    tau_hist, hi_hist = tau_hist[order], hi_hist[order]
    return predict_rul_many(
        prior, tau_hist, hi_hist, np.array([tau_now]), np.array([len(tau_hist)]), horizon_s,
        draws, seed,
    )[0]
