"""Signal definitions. Every function returns a panel indexed at the DECISION date T, built only from
EOD data through T-1 and the day-T 3PM snapshot. Positive values mean "expected to outperform".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.hazard import CurveFit, fit_curve_panel
from src.research.panels import Panels, hedged_return, rates_return
from src.research.stats import trailing_z
from src.research.universe import CREDIT_REFERENCE

MAX_ABS_BASIS_BP = 1000.0


# --------------------------------------------------------------------------- NAV basis (lecture 2)
def nav_basis_bp(close: pd.DataFrame, nav: pd.DataFrame) -> pd.DataFrame:
    """ETF basis = close / NAV - 1 in bp, dated at the close it is observed on. |basis| > 1000bp -> NaN."""
    b = (close / nav - 1.0) * 1e4
    return b.where(b.abs() < MAX_ABS_BASIS_BP)


def nav_discount_lagged(p: Panels, window: int = 60) -> pd.DataFrame:
    """Discount to NAV at the T-1 close as a z-score against its own trailing history (cheap = positive)."""
    return (-trailing_z(nav_basis_bp(p.close, p.nav), window, window // 2)).shift(1)


def nav_discount_3pm(p: Panels, window: int = 60) -> pd.DataFrame:
    """Discount to NAV implied by the day-T 3PM mid, as a z-score (cheap = positive).

    The 3PM price is compared with the T-1 NAV after removing the rates move since the T-1 close:
        basis_3pm = (1 + basis_{T-1}) (1 + r_{T-1 close -> 3PM}) - 1 - rates return to 3PM.
    The total-return form of the 3PM move keeps ex-dividend dates from showing up as a discount.
    """
    basis_prev = (p.close / p.nav - 1.0).shift(1)
    b3 = ((1.0 + basis_prev) * (1.0 + p.ret_to_3pm_mid) - 1.0 - rates_return(p.ret_to_3pm_mid, p.a("duration"))) * 1e4
    b3 = b3.where(b3.abs() < MAX_ABS_BASIS_BP)
    return -trailing_z(b3, window, window // 2)


# --------------------------------------------------------------------------- 3PM factor residual
def reference_beta(p: Panels, window: int = 120, reference: dict[str, str] | None = None) -> pd.DataFrame:
    """Rolling beta of each credit ETF's rates-hedged daily return on its bucket's reference ETF,
    estimated through T-1 (lagged one row). Columns: ETFs that have a reference in the panel."""
    reference = CREDIT_REFERENCE if reference is None else reference
    xr = hedged_return(p.ret, p.a("duration"))
    out = {}
    for t in p.tickers:
        ref = reference.get(p.bucket[t])
        if ref is None or ref not in xr or t == ref:
            continue
        cov = xr[t].rolling(window, min_periods=window // 2).cov(xr[ref])
        out[t] = (cov / xr[ref].rolling(window, min_periods=window // 2).var()).shift(1)
    return pd.DataFrame(out)


def factor_residual(p: Panels, leg: pd.DataFrame, beta: pd.DataFrame, reference: dict[str, str] | None = None) -> pd.DataFrame:
    """`leg` return minus its rates part minus beta x the reference ETF's hedged return over the same leg."""
    reference = CREDIT_REFERENCE if reference is None else reference
    x = hedged_return(leg, p.a("duration"))
    return pd.DataFrame({t: x[t] - beta[t] * x[reference[p.bucket[t]]] for t in beta.columns})


def residual_3pm_reversal(p: Panels, window: int = 120) -> pd.DataFrame:
    """Minus the part of the move from the T-1 close to 3PM on T that rates and the reference credit ETF
    do not explain (an ETF that ran ahead of its factors gets a negative score)."""
    return -factor_residual(p, p.ret_to_3pm_mid, reference_beta(p, window))


# --------------------------------------------------------------------------- hazard-curve edge (lecture 5)
def liquidity_weights(p: Panels, tickers: list[str]) -> pd.DataFrame:
    """Calibration weights ~ liquidity / DV01 (lecture 5): sqrt(trailing median $ volume) / duration."""
    dollar = (p.volume * p.close)[tickers].rolling(60, min_periods=20).median()
    return np.sqrt(dollar) / p.a("duration")[tickers]


def hazard_edge_fit(p: Panels, tickers: list[str], lam: float = 1.0, n_terms: int = 3) -> CurveFit:
    """Nelson-Siegel fit of g-spread against average maturity across `tickers`, on each close.
    Edges are dated at the close they are observed on; lag them one day before use."""
    return fit_curve_panel(p.a("gspread")[tickers], p.a("avgttm")[tickers], liquidity_weights(p, tickers), lam, n_terms)


def edge_deviation_z(edges: pd.DataFrame, window: int = 120) -> pd.DataFrame:
    """Edge relative to the fund's own trailing mean, in trailing standard deviations, known at T
    (wide to its own history = positive). The mean is removed because funds differ in credit quality."""
    return trailing_z(edges, window, window // 3).shift(1)


# --------------------------------------------------------------------------- roll-down (lecture 1)
def rolldown_pct(curve: CurveFit, ttm: pd.DataFrame, duration: pd.DataFrame, per_bp: bool = False) -> pd.DataFrame:
    """Roll-down return in % of price per year: duration x slope of the fitted curve at the ETF's maturity.

    With an unchanged curve a bond one year shorter is valued at a level lower by the slope, so its price
    rises by duration x slope. `curve` must be fitted on values in % (set per_bp=True if it is in bp).
    """
    slope = curve.slope(ttm)
    return duration.reindex_like(slope) * slope * (0.01 if per_bp else 1.0)
