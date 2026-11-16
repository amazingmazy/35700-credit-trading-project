"""Hazard-rate tools (lectures 4-5): the credit triangle and a Nelson-Siegel curve fitted across ETFs.

Each ETF is treated as one synthetic bond with maturity `avgttm` and spread `gspread`. With the credit
triangle s = (1 - R) h (lecture 4), a smooth average-hazard curve H(T) implies a spread curve
s(T) = (1 - R) H(T). The curve is fitted across the ETFs of a tier each day by weighted least squares;
market spread minus fitted spread is the "model edge" of lecture 5.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def ns_loadings(ttm: np.ndarray, lam: float, n_terms: int = 3) -> np.ndarray:
    """Nelson-Siegel loadings at maturities `ttm` (years): level, slope and (if n_terms = 3) curvature.

    f1(x) = (1 - e^-x) / x,  f2(x) = f1(x) - e^-x,  x = ttm / lam.
    """
    if n_terms not in (1, 2, 3):
        raise ValueError("n_terms must be 1, 2 or 3")
    x = np.asarray(ttm, dtype=float) / lam
    f1 = (1.0 - np.exp(-x)) / x
    cols = [np.ones_like(x), f1, f1 - np.exp(-x)]
    return np.column_stack(cols[:n_terms])


def fit_ns(values: np.ndarray, ttm: np.ndarray, weights: np.ndarray, lam: float, n_terms: int = 3) -> np.ndarray:
    """Weighted least-squares Nelson-Siegel coefficients for one cross-section."""
    X = ns_loadings(ttm, lam, n_terms)
    sw = np.sqrt(np.asarray(weights, dtype=float))
    theta, *_ = np.linalg.lstsq(X * sw[:, None], np.asarray(values, dtype=float) * sw, rcond=None)
    return theta


@dataclass
class CurveFit:
    edges: pd.DataFrame  # market - fitted, same units as the input values
    fitted: pd.DataFrame
    theta: pd.DataFrame  # coefficients per date
    rmse: pd.Series  # weighted RMSE per date
    lam: float
    n_terms: int

    def slope(self, ttm: pd.DataFrame, step: float = 0.25) -> pd.DataFrame:
        """d(fitted value)/d(maturity) at each ETF's maturity, by a backward difference of `step` years."""
        out = np.full(ttm.shape, np.nan)
        th, t = self.theta.reindex(ttm.index).to_numpy(), ttm.to_numpy(dtype=float)
        for i in range(len(ttm)):
            m = np.isfinite(t[i]) & (t[i] > step + 0.05)
            if not np.isfinite(th[i]).all() or not m.any():
                continue
            hi = ns_loadings(t[i][m], self.lam, self.n_terms) @ th[i]
            lo = ns_loadings(t[i][m] - step, self.lam, self.n_terms) @ th[i]
            out[i, m] = (hi - lo) / step
        return pd.DataFrame(out, index=ttm.index, columns=ttm.columns)


def fit_curve_panel(
    values: pd.DataFrame,
    ttm: pd.DataFrame,
    weights: pd.DataFrame | None = None,
    lam: float = 1.0,
    n_terms: int = 3,
    min_names: int | None = None,
) -> CurveFit:
    """Fit a Nelson-Siegel curve across the columns on every date.

    A date is skipped when fewer than `min_names` (default n_terms + 3) columns have a value, a maturity
    above 0.05 years and a positive weight.
    """
    cols = values.columns
    ttm = ttm.reindex(index=values.index, columns=cols)
    w = pd.DataFrame(1.0, index=values.index, columns=cols) if weights is None else weights.reindex(index=values.index, columns=cols)
    need = n_terms + 3 if min_names is None else min_names
    V, T, W = values.to_numpy(dtype=float), ttm.to_numpy(dtype=float), w.to_numpy(dtype=float)
    fitted = np.full(V.shape, np.nan)
    theta = np.full((len(values), n_terms), np.nan)
    rmse = np.full(len(values), np.nan)
    for i in range(len(values)):
        m = np.isfinite(V[i]) & np.isfinite(T[i]) & np.isfinite(W[i]) & (T[i] > 0.05) & (W[i] > 0)
        if m.sum() < need:
            continue
        wi = W[i][m] / W[i][m].mean()
        th = fit_ns(V[i][m], T[i][m], wi, lam, n_terms)
        fit = ns_loadings(T[i][m], lam, n_terms) @ th
        fitted[i, m], theta[i] = fit, th
        rmse[i] = float(np.sqrt(np.average((V[i][m] - fit) ** 2, weights=wi)))
    fitted = pd.DataFrame(fitted, index=values.index, columns=cols)
    return CurveFit(values - fitted, fitted, pd.DataFrame(theta, index=values.index), pd.Series(rmse, index=values.index), lam, n_terms)


def implied_hazard(spread_bp: pd.DataFrame | pd.Series | float, recovery: float):
    """Credit triangle: h = s / (1 - R), in the units of the spread."""
    if not 0.0 <= recovery < 1.0:
        raise ValueError("recovery must be in [0, 1)")
    return spread_bp / (1.0 - recovery)


def equalizing_recovery(spread_leg_bp, hazard_ref_bp):
    """Recovery R that makes the leg's implied hazard equal the reference hazard: R = 1 - s_leg / h_ref."""
    return 1.0 - spread_leg_bp / hazard_ref_bp


def log_hazard_gap(spread_leg_bp: pd.Series, spread_ref_bp: pd.Series, recovery_leg: float, recovery_ref: float) -> pd.Series:
    """log(h_leg / h_ref); positive when the leg prices a higher hazard than the reference."""
    h_leg, h_ref = implied_hazard(spread_leg_bp, recovery_leg), implied_hazard(spread_ref_bp, recovery_ref)
    return np.log(h_leg.where(h_leg > 0) / h_ref.where(h_ref > 0))
