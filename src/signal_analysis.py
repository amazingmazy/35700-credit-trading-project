"""Signal diagnostics: IC, Rank IC, ICIR and IC decay.

Alignment: the signal for date T is formed at 3:45PM on T and trades at close_T, so it is
scored against the forward total return from close_T to close_{T+h} (split & dividend adjusted).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def forward_returns(total_return: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Compounded total return from close_T to close_{T+h}, indexed at T."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    growth = np.log1p(total_return)
    fwd = growth.rolling(horizon, min_periods=horizon).sum().shift(-horizon)
    return np.expm1(fwd)


def _row_corr(a: pd.DataFrame, b: pd.DataFrame, min_obs: int) -> pd.Series:
    a, b = a.align(b, join="inner")
    valid = a.notna() & b.notna()
    a, b = a.where(valid), b.where(valid)
    n = valid.sum(axis=1)
    am = a.sub(a.mean(axis=1), axis=0)
    bm = b.sub(b.mean(axis=1), axis=0)
    cov = (am * bm).sum(axis=1)
    denom = np.sqrt((am**2).sum(axis=1) * (bm**2).sum(axis=1))
    ic = cov / denom.replace(0, np.nan)
    return ic.where(n >= min_obs)


def information_coefficient(signal: pd.DataFrame, fwd: pd.DataFrame, rank: bool = False, min_obs: int = 5) -> pd.Series:
    """Daily cross-sectional Pearson (rank=False) or Spearman (rank=True) correlation."""
    s, f = signal.align(fwd, join="inner")
    valid = s.notna() & f.notna()
    s, f = s.where(valid), f.where(valid)
    if rank:
        s, f = s.rank(axis=1), f.rank(axis=1)
    return _row_corr(s, f, min_obs).dropna()


def ic_summary(ic: pd.Series, horizon: int = 1, periods_per_year: int = 252) -> dict:
    ic = ic.dropna()
    n = len(ic)
    mean = float(ic.mean()) if n else float("nan")
    sd = float(ic.std(ddof=1)) if n > 1 else float("nan")
    icir = mean / sd if n > 1 and sd > 0 else float("nan")
    return {
        "mean": mean,
        "std": sd,
        "icir": icir,
        "icir_annualized": icir * math.sqrt(periods_per_year / horizon) if np.isfinite(icir) else float("nan"),
        "t_stat": icir * math.sqrt(n) if np.isfinite(icir) else float("nan"),
        "hit_rate": float((ic > 0).mean()) if n else float("nan"),
        "n_days": n,
    }


class SignalAnalyzer:
    def __init__(self, total_return: pd.DataFrame, periods_per_year: int = 252, min_obs: int = 5):
        self.total_return = total_return
        self.periods_per_year = periods_per_year
        self.min_obs = min_obs

    def analyze(self, signal: pd.DataFrame, horizons: tuple[int, ...] = (1, 5, 20)) -> pd.DataFrame:
        """Rows: (metric, horizon) -> IC and Rank IC summaries."""
        rows = []
        for h in horizons:
            fwd = forward_returns(self.total_return, h)
            for kind, rank in (("IC", False), ("RankIC", True)):
                ic = information_coefficient(signal, fwd, rank=rank, min_obs=self.min_obs)
                rows.append({"type": kind, "horizon": h, **ic_summary(ic, h, self.periods_per_year)})
        return pd.DataFrame(rows).set_index(["type", "horizon"])

    def ic_series(self, signal: pd.DataFrame, horizon: int = 1, rank: bool = False) -> pd.Series:
        return information_coefficient(signal, forward_returns(self.total_return, horizon), rank, self.min_obs)
