"""Performance metrics.

Required metrics follow project statement section 5 exactly:
  r_T = PnL_T / initial_capital (fixed, no compounding), rf = 0, ddof = 1, 252 obs/yr,
  V_T = C0 + cumsum(PnL), M_T = max(V_0..V_T) with V_0 = C0, all trading days included.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.config import BacktestConfig

if TYPE_CHECKING:  # pragma: no cover
    from src.backtester import BacktestResult


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252, rf: float = 0.0) -> float:
    r = pd.Series(returns, dtype=float).dropna() - rf
    if len(r) < 2:
        return float("nan")
    sd = r.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return float("nan")
    return float(math.sqrt(periods_per_year) * r.mean() / sd)


def equity_curve(pnl: pd.Series, initial_capital: float) -> pd.Series:
    return initial_capital + pd.Series(pnl, dtype=float).fillna(0.0).cumsum()


def drawdowns(pnl: pd.Series, initial_capital: float) -> tuple[pd.Series, pd.Series]:
    """($ drawdown series, % drawdown series), positive loss magnitudes, with V_0 in the running max."""
    v = equity_curve(pnl, initial_capital)
    m = np.maximum(v.cummax(), initial_capital)
    return m - v, 1.0 - v / m


def max_drawdown(pnl: pd.Series, initial_capital: float) -> tuple[float, float]:
    if len(pnl) == 0:
        return 0.0, 0.0
    dd_d, dd_p = drawdowns(pnl, initial_capital)
    return float(max(dd_d.max(), 0.0)), float(max(dd_p.max(), 0.0))


def information_ratio(returns: pd.Series, benchmark: pd.Series, periods_per_year: int = 252) -> float:
    active = (pd.Series(returns, dtype=float) - pd.Series(benchmark, dtype=float).reindex(returns.index)).dropna()
    return sharpe_ratio(active, periods_per_year)


class PerformanceAnalyzer:
    def __init__(self, config: BacktestConfig | None = None):
        self.config = config or BacktestConfig()

    def required_metrics(self, pnl: pd.Series) -> dict:
        c = self.config
        r = pnl / c.initial_capital
        dd_d, dd_p = max_drawdown(pnl, c.initial_capital)
        return {
            "n_days": int(len(pnl)),
            "total_net_pnl": float(pnl.sum()),
            "annualized_net_pnl": float(c.periods_per_year * pnl.mean()) if len(pnl) else float("nan"),
            "annualized_return": float(c.periods_per_year * r.mean()) if len(r) else float("nan"),
            "annualized_vol": float(math.sqrt(c.periods_per_year) * r.std(ddof=1)) if len(r) > 1 else float("nan"),
            "sharpe": sharpe_ratio(r, c.periods_per_year, c.risk_free_rate),
            "max_drawdown_usd": dd_d,
            "max_drawdown_pct": dd_p,
        }

    def summary(self, result: "BacktestResult", benchmark_returns: pd.Series | None = None) -> dict:
        c = self.config
        d = result.daily
        out = self.required_metrics(d["pnl"])
        gross_pnl = d["pnl"] + d["costs"]
        active = d["traded_shares"] > 0
        out.update({
            "warmup_days": int(d["warmup"].sum()),
            "gross_sharpe": sharpe_ratio(gross_pnl / c.initial_capital, c.periods_per_year),
            "total_costs": float(d["costs"].sum()),
            "annualized_costs": float(c.periods_per_year * d["costs"].mean()),
            "total_dividends": float(d["dividends"].sum()),
            "avg_daily_turnover": float(d["turnover"].mean()),
            "avg_gmv": float(d["gmv"].mean()),
            "max_gmv": float(d["gmv"].max()),
            "avg_gmv_utilisation": float(d["gmv"].mean() / c.gmv_cap),
            "gmv_breach_days": int(d["gmv_breach"].sum()),
            "avg_dv01": float(d["dv01"].mean()),
            "max_abs_dv01": float(d["dv01"].abs().max()),
            "hit_rate": float((d["pnl"] > 0).sum() / max((d["pnl"] != 0).sum(), 1)),
            "trading_days": int(active.sum()),
        })
        if benchmark_returns is not None:
            out["information_ratio"] = information_ratio(d["return"], benchmark_returns, c.periods_per_year)
            out["benchmark_sharpe"] = sharpe_ratio(benchmark_returns.reindex(d.index), c.periods_per_year)
        return out

    @staticmethod
    def format_summary(metrics: dict) -> str:
        lines = []
        for k, v in metrics.items():
            if isinstance(v, float):
                if any(s in k for s in ("pct", "utilisation", "rate", "turnover", "_return", "_vol")):
                    lines.append(f"{k:<24}{v:>18.2%}")
                elif abs(v) >= 1000:
                    lines.append(f"{k:<24}{v:>18,.0f}")
                else:
                    lines.append(f"{k:<24}{v:>18.4f}")
            else:
                lines.append(f"{k:<24}{v!s:>18}")
        return "\n".join(lines)
