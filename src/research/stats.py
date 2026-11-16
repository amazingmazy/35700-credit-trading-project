"""Signal statistics: z-scores, half-life, forward returns, rank IC with overlap-aware t-stats,
and the net-of-financing Sharpe."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.signal_analysis import newey_west_tstat


def trailing_z(x: pd.DataFrame | pd.Series, window: int = 120, min_periods: int = 40):
    """(x - trailing mean) / trailing std, where both statistics end the row BEFORE the current one."""
    mean = x.rolling(window, min_periods=min_periods).mean().shift(1)
    std = x.rolling(window, min_periods=min_periods).std().shift(1)
    return (x - mean) / std.where(std > 0)


def half_life(x: pd.Series, min_obs: int = 100) -> float:
    """Half-life in rows implied by the lag-1 autocorrelation (NaN if it is not in (0, 1))."""
    x = pd.Series(x, dtype=float).dropna()
    if len(x) < min_obs:
        return float("nan")
    phi = x.autocorr(1)
    return float(-math.log(2.0) / math.log(phi)) if 0.0 < phi < 1.0 else float("nan")


def forward_sum(ret: pd.DataFrame | pd.Series, horizon: int = 1):
    """Sum of the next `horizon` rows of `ret` (rows T+1 .. T+horizon), indexed at T."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    return ret.rolling(horizon, min_periods=horizon).sum().shift(-horizon)


def demean_by_group(panel: pd.DataFrame, groups: pd.Series | None) -> pd.DataFrame:
    """Subtract the cross-sectional mean (within each group if `groups` maps column -> group)."""
    if groups is None:
        return panel.sub(panel.mean(axis=1), axis=0)
    g = pd.Series(groups).reindex(panel.columns)
    return panel - panel.T.groupby(g.to_numpy()).transform("mean").T


def rank_ic(signal: pd.DataFrame, fwd: pd.DataFrame, groups: pd.Series | None = None, min_obs: int = 5) -> pd.Series:
    """Daily cross-sectional Spearman correlation of signal and forward return over common valid names.

    With `groups`, both are demeaned within group first, so the IC measures ranking inside buckets.
    """
    s, f = signal.align(fwd, join="inner")
    valid = s.notna() & f.notna()
    s, f = s.where(valid), f.where(valid)
    if groups is not None:
        s, f = demean_by_group(s, groups), demean_by_group(f, groups)
    ok = valid.sum(axis=1) >= min_obs
    return s[ok].rank(axis=1).corrwith(f[ok].rank(axis=1), axis=1).dropna()


def ic_table(
    signal: pd.DataFrame,
    ret: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 5, 20),
    groups: pd.Series | None = None,
    min_obs: int = 5,
) -> pd.DataFrame:
    """Mean rank IC per horizon with a Newey-West t-stat (horizon-1 lags) and the number of days.

    `signal` is indexed at the decision date T; `ret` is the daily return panel to be summed forward.
    """
    rows = []
    for h in horizons:
        ic = rank_ic(signal, forward_sum(ret, h), groups, min_obs)
        rows.append({"horizon": h, "mean_ic": ic.mean() if len(ic) else np.nan,
                     "t_stat": newey_west_tstat(ic, h - 1), "n_days": len(ic)})
    return pd.DataFrame(rows).set_index("horizon")


def predict_ts(signal: pd.Series, ret: pd.Series, horizon: int = 1) -> dict:
    """Time-series relation of signal(T) with the sum of `ret` over T+1..T+horizon.

    Returns the correlation, the OLS slope and a Newey-West t-stat of the slope with horizon-1 lags
    (overlapping forward windows make the regression errors serially correlated).
    """
    df = pd.concat([pd.Series(signal, dtype=float), forward_sum(pd.Series(ret, dtype=float), horizon)], axis=1).dropna()
    n = len(df)
    if n < 10:
        return {"corr": np.nan, "slope": np.nan, "t_stat": np.nan, "n": n}
    x = df.iloc[:, 0].to_numpy() - df.iloc[:, 0].mean()
    y = df.iloc[:, 1].to_numpy() - df.iloc[:, 1].mean()
    sxx = float(x @ x)
    if not sxx > 0:
        return {"corr": np.nan, "slope": np.nan, "t_stat": np.nan, "n": n}
    beta = float(x @ y) / sxx
    g = x * (y - beta * x)
    s = float(g @ g)
    lags = horizon - 1
    for lag in range(1, min(lags, n - 1) + 1):
        s += 2.0 * (1.0 - lag / (lags + 1.0)) * float(g[lag:] @ g[:-lag])
    t = beta / math.sqrt(s) * sxx if s > 0 else np.nan
    return {"corr": float(np.corrcoef(x, y)[0, 1]), "slope": beta, "t_stat": float(t), "n": n}


def sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = pd.Series(returns, dtype=float).dropna()
    sd = r.std(ddof=1)
    if len(r) < 2 or not sd > 0:
        return float("nan")
    return float(math.sqrt(periods_per_year) * r.mean() / sd)


def financing_adjusted_sharpe(
    returns: pd.Series,
    net_exposure: pd.Series | float,
    bill_rate: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Sharpe after charging the T-bill rate on the net exposure.

    `returns` and `net_exposure` are in the same units (per $1 of capital). `net_exposure` dated T is the
    net long exposure carried INTO day T; `bill_rate` is annualized (decimal). For net_exposure = 0 this
    equals the plain Sharpe.
    """
    r = pd.Series(returns, dtype=float)
    expo = pd.Series(net_exposure, index=r.index, dtype=float) if np.isscalar(net_exposure) else pd.Series(net_exposure, dtype=float).reindex(r.index)
    rate = pd.Series(bill_rate, dtype=float).reindex(r.index).ffill()
    return sharpe(r - expo * rate / periods_per_year, periods_per_year)
