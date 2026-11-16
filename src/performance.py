"""Performance metrics.

Required metrics follow project statement section 5 exactly:
  r_T = PnL_T / initial_capital (fixed, no compounding), rf = 0, ddof = 1, 252 obs/yr,
  V_T = C0 + cumsum(PnL), M_T = max(V_0..V_T) with V_0 = C0, all trading days included.

Working metric (group rule): `sharpe_net_of_financing` charges the T-bill rate on the book's net $
exposure. The project-formula `sharpe` treats the full yield of a net-long book as excess return
(a T-bill ETF, which earns only the risk-free rate, scores far above 1 on it), so it is reported only
next to the net-of-financing figure, and any net-of-financing Sharpe above SHARPE_AUDIT_THRESHOLD is
flagged for an audit.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.config import BacktestConfig

if TYPE_CHECKING:  # pragma: no cover
    from src.backtester import BacktestResult


SHARPE_AUDIT_THRESHOLD = 1.0  # a Sharpe above this (net of financing) is audited before it is reported


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252, rf: float = 0.0) -> float:
    """Annualized Sharpe of per-period returns. `rf` is an ANNUALIZED rate: rf / periods_per_year is
    subtracted from each return. The project statement fixes rf = 0."""
    r = pd.Series(returns, dtype=float).dropna() - rf / periods_per_year
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


def bill_rate_from_market(market, ticker: str = "SHV", lag: int = 1) -> pd.Series | None:
    """Annualized T-bill rate (decimal) per date, taken from the `yield` field (%) of a T-bill ETF.

    The value is lagged by `lag` days and forward-filled; the leading gap takes the first valid value.
    Returns None if the ETF or the field is not in the data. Used only for reporting.
    """
    if "yield" not in market.eod:
        return None
    try:
        sym = market.symbol_for(ticker)
    except KeyError:
        return None
    if sym not in market.eod["yield"].columns:
        return None
    y = market.eod["yield"][sym]
    y = y.where(y.between(-1.0, 15.0)).ffill().shift(lag).bfill()
    if y.isna().all():
        return None
    return y / 100.0


def financing_charge(net_exposure: pd.Series, bill_rate: pd.Series, periods_per_year: int = 252) -> pd.Series:
    """$ financing cost booked on day T = net $ exposure carried from the T-1 close x bill rate / periods.

    Net exposure = long market value + short market value (shorts negative). Holding it instead of
    cash gives up the bill rate on that amount; the project rules set this cost to zero.
    """
    carried = pd.Series(net_exposure, dtype=float).shift(1).fillna(0.0)
    rate = pd.Series(bill_rate, dtype=float).reindex(carried.index).ffill().fillna(0.0)
    return carried * rate / periods_per_year


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

    def financing_adjusted(self, daily: pd.DataFrame, bill_rate: pd.Series) -> dict:
        """Working metric: the same PnL after charging the T-bill rate on the net $ exposure.

        `sharpe` (project formula: r_f = 0, zero financing) counts the full yield of a net-long book as
        excess return. This removes that part. For a dollar-neutral book the two are equal.
        """
        c = self.config
        net = daily["long_mv"] + daily["short_mv"]
        charge = financing_charge(net, bill_rate, c.periods_per_year)
        pnl_net = daily["pnl"] - charge
        s_net = sharpe_ratio(pnl_net / c.initial_capital, c.periods_per_year)
        return {
            "avg_net_exposure": float(net.mean()),
            "annualized_financing_charge": float(c.periods_per_year * charge.mean()),
            "annualized_pnl_net_of_financing": float(c.periods_per_year * pnl_net.mean()),
            "sharpe_net_of_financing": s_net,
            "sharpe_audit_required": bool(np.isfinite(s_net) and s_net > SHARPE_AUDIT_THRESHOLD),
        }

    def summary(
        self,
        result: "BacktestResult",
        benchmark_returns: pd.Series | None = None,
        bill_rate: pd.Series | None = None,
    ) -> dict:
        c = self.config
        d = result.daily
        out = self.required_metrics(d["pnl"])
        if bill_rate is not None:
            out.update(self.financing_adjusted(d, bill_rate))
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
        labels = {"sharpe": "sharpe_project_formula", "gross_sharpe": "gross_sharpe_project"}
        lines = []
        for k, v in metrics.items():
            k = labels.get(k, k)
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
