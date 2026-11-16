"""Merton (1974) structural model applied to a bond ETF (lecture 6).

Setup: assets A follow a lognormal diffusion with volatility sigma_A; debt is a zero-coupon claim with
face K due at T; leverage L = K / A. Then, with
    d1 = (-ln L + (r + sigma_A^2 / 2) T) / (sigma_A sqrt(T)),   d2 = d1 - sigma_A sqrt(T),
    B / A   = Phi(-d1) + L e^{-rT} Phi(d2)                        (risky debt = assets - call)
    spread  = -ln( e^{rT} Phi(-d1) / L + Phi(d2) ) / T            (lecture 6, credit spread)
    sigma_B = Phi(-d1) / (B / A) * sigma_A                        (Ito on B(A): dB/dA = Phi(-d1))
The slides give the equity-vol version of the last line; the debt-vol version follows the same way.

A bond ETF supplies two observables, its credit spread and (via its listed options) the volatility of
the debt claim, so (L, sigma_A) can be solved each day. The fit is to a diversified portfolio, not a
firm: L and sigma_A are effective index-level parameters.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq, fsolve
from scipy.stats import norm


def merton_debt(L, sigma_a, T, r):
    """(B/A, credit spread, debt volatility) for leverage L = K/A. Works on scalars or arrays."""
    q = sigma_a * np.sqrt(T)
    d1 = (-np.log(L) + (r + 0.5 * sigma_a**2) * T) / q
    d2 = d1 - q
    b_over_a = norm.cdf(-d1) + L * np.exp(-r * T) * norm.cdf(d2)
    spread = -np.log(b_over_a * np.exp(r * T) / L) / T
    debt_vol = norm.cdf(-d1) / b_over_a * sigma_a
    return b_over_a, spread, debt_vol


def solve_leverage_vol(spread: float, debt_vol: float, T: float, r: float, x0=(0.2, -2.3)):
    """Solve (L, sigma_A) from a spread and a debt volatility (both decimals). Returns (nan, nan) on failure.

    `x0` is the starting point in (ln L, ln sigma_A).
    """
    target = np.array([spread, debt_vol], dtype=float)
    if not (np.isfinite(target).all() and np.isfinite([T, r]).all()) or spread <= 0 or debt_vol <= 0 or T <= 0:
        return float("nan"), float("nan")

    def resid(x):
        _, s, v = merton_debt(np.exp(x[0]), np.exp(x[1]), T, r)
        return np.array([s, v]) / target - 1.0

    with np.errstate(all="ignore"):
        x, _, ok, _ = fsolve(resid, np.asarray(x0, dtype=float), full_output=True)
        good = ok == 1 and np.isfinite(x).all() and np.abs(resid(x)).max() < 1e-6
    return (float(np.exp(x[0])), float(np.exp(x[1]))) if good else (float("nan"), float("nan"))


def solve_panel(spread: pd.Series, debt_vol: pd.Series, T: pd.Series, r: pd.Series) -> pd.DataFrame:
    """Day-by-day solve of (L, sigma_A); each day starts from the previous day's solution."""
    df = pd.concat([spread, debt_vol, T, r], axis=1, keys=["s", "v", "T", "r"])
    out = np.full((len(df), 2), np.nan)
    x0 = (0.2, -2.3)
    for i, row in enumerate(df.itertuples(index=False)):
        L, sig = solve_leverage_vol(row.s, row.v, row.T, row.r, x0)
        if np.isfinite(L):
            out[i] = (L, sig)
            x0 = (np.log(L), np.log(sig))
    return pd.DataFrame(out, index=df.index, columns=["L", "sigma_a"])


def implied_leverage(spread: float, sigma_a: float, T: float, r: float) -> float:
    """Leverage that reproduces `spread` for a given asset volatility (the spread rises with L)."""
    if not np.isfinite([spread, sigma_a, T, r]).all() or spread <= 0 or sigma_a <= 0 or T <= 0:
        return float("nan")
    f = lambda x: merton_debt(np.exp(x), sigma_a, T, r)[1] - spread
    try:
        return float(np.exp(brentq(f, -4.0, 4.0)))
    except ValueError:
        return float("nan")


def implied_leverage_panel(spread: pd.Series, sigma_a: pd.Series, T: pd.Series, r: pd.Series) -> pd.Series:
    df = pd.concat([spread, sigma_a, T, r], axis=1, keys=["s", "v", "T", "r"])
    return pd.Series([implied_leverage(x.s, x.v, x.T, x.r) for x in df.itertuples(index=False)], index=df.index)


_Z, _W = np.polynomial.hermite_e.hermegauss(60)
_W = _W / _W.sum()


def debt_option_implied_vol(L: float, sigma_a: float, T: float, r: float, tau: float) -> float:
    """Black implied vol of an at-the-money-forward option, expiring in `tau` years, on the debt claim.

    The debt is revalued at expiry with an unchanged maturity T (a fund that keeps its maturity
    constant). The expectation over the lognormal asset value is taken by Gauss-Hermite quadrature,
    as lecture 6 does for equity options ("1-dimensional integration ... vs the log-normal density").
    """
    if not np.isfinite([L, sigma_a, T, r, tau]).all() or tau <= 0:
        return float("nan")
    growth = np.exp((r - 0.5 * sigma_a**2) * tau + sigma_a * np.sqrt(tau) * _Z)  # A_tau / A_0
    b = growth * merton_debt(L / growth, sigma_a, T, r)[0]  # B_tau / A_0
    fwd = float(_W @ b)
    call = float(_W @ np.maximum(b - fwd, 0.0))
    p = (call / fwd + 1.0) / 2.0
    if not 0.5 < p < 1.0:
        return float("nan")
    return float(2.0 * norm.ppf(p) / np.sqrt(tau))


def rates_vol(etf_duration: pd.Series, anchor_vol: pd.DataFrame, anchor_duration: pd.DataFrame) -> pd.Series:
    """Price vol the ETF would have from rates alone: its duration x the Treasury ETFs' option-implied
    yield vol (vol / duration), interpolated in duration. Same units as `anchor_vol`."""
    yv = (anchor_vol / anchor_duration).to_numpy(dtype=float)
    xa = anchor_duration.to_numpy(dtype=float)
    d = etf_duration.reindex(anchor_vol.index).to_numpy(dtype=float)
    out = np.full(len(d), np.nan)
    for i in range(len(d)):
        if np.isfinite(yv[i]).all() and np.isfinite(xa[i]).all() and np.isfinite(d[i]):
            order = np.argsort(xa[i])
            out[i] = np.interp(d[i], xa[i][order], yv[i][order]) * d[i]
    return pd.Series(out, index=anchor_vol.index)


def credit_only_vol(total_vol: pd.Series, rates_only_vol: pd.Series, rho: pd.Series | float) -> pd.Series:
    """Credit component c of an ETF's price vol, from total^2 = rates^2 + c^2 + 2 rho rates c.

    `rho` is the correlation between rate changes and spread changes. NaN where no real solution exists
    (the ETF's implied vol is below what rates alone would give).
    """
    disc = (rho * rates_only_vol) ** 2 + total_vol**2 - rates_only_vol**2
    return -rho * rates_only_vol + np.sqrt(disc.where(disc > 0))
