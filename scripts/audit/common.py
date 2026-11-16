"""Shared helpers for the audit scripts in this folder: paper books and engine runs on the clean ETFs.

Run a script from anywhere, e.g. `python scripts/audit/paper_books.py`. The four course parquet files must be
in `data/`. Tables are printed and also written to `results/audit/` (not committed).

Conventions used by every script here (see docs/results_audit.md):
  * A signal dated T is known at the day-T decision (EOD data through T-1, or the day-T 3PM snapshot).
  * A book decided at T is entered at the close of T and earns the return of T+1.
  * A missing signal is carried forward for up to 10 days and every fund is then re-ranked, so a book
    always nets to zero. A day with too few funds keeps the whole previous book.
  * Commission is $0.0035 per share on the fund legs and on the Treasury hedge legs.
  * Financing: the T-bill rate (SHV yield field, lagged a day) on the net dollars of all legs.
"""
from __future__ import annotations

import pathlib
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from src.backtester import Backtester  # noqa: E402
from src.config import BacktestConfig  # noqa: E402
from src.data_loader import DataLoader  # noqa: E402
from src.performance import bill_rate_from_market, financing_charge  # noqa: E402
from src.research import hazard, merton, signals, stats  # noqa: E402,F401
from src.research.panels import build_panels, hedged_return, rates_return  # noqa: E402,F401
from src.research.universe import HEDGE_ANCHORS, RECOVERY, in_bucket  # noqa: E402,F401
from src.strategies.base import WeightStrategy  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 300)
pd.set_option("display.max_columns", 40)

OUT = ROOT / "results" / "audit"
OUT.mkdir(parents=True, exist_ok=True)
COST = 0.0035  # $ per share traded
CARRY_DAYS = 10  # a missing signal is carried forward for at most this many days

market = DataLoader(ROOT / "data").load()
p = build_panels(market)
dur, gs, ttm, ylds = p.a("duration"), p.a("gspread"), p.a("avgttm"), p.a("yield")
bill = bill_rate_from_market(market)
anch = list(HEDGE_ANCHORS)
hy, ig, ust = in_bucket(p.tickers, "HY"), in_bucket(p.tickers, "IG"), in_bucket(p.tickers, "UST")
credit = in_bucket(p.tickers, "IG", "FRN", "HY", "HY-Global", "Loan")
cap_mm = 0.03 * (p.volume * p.close).median() / 1e6  # 3% of median daily dollar volume, $MM
liquid = [t for t in credit if cap_mm[t] >= 1.0]
groups = p.bucket
tick_idx = {t: i for i, t in enumerate(dur.columns)}
sym = {t: market.symbol_for(t) for t in p.tickers}
dlag = dur.ffill().shift(1)  # duration known at the decision for the next day's return
xr = hedged_return(p.ret, dur)  # rates-hedged daily return


# ----------------------------------------------------------------------------- hedge legs
def anchor_dollars(durp: pd.DataFrame) -> np.ndarray:
    """[date, ticker, anchor]: $ of each Treasury anchor shorted per $1 of the fund in the duration-matched
    hedge. Row T uses durations through T-1 (the same interpolation as `rates_return`)."""
    dl = durp.ffill().shift(1)
    xa, di = dl[anch].to_numpy(), dl.to_numpy()
    out = np.full(di.shape + (len(anch),), np.nan)
    for i in range(len(dl)):
        if np.isfinite(xa[i]).all():
            o = np.argsort(xa[i])
            xs = xa[i][o]
            for k in range(len(anch)):
                e = np.zeros(len(anch))
                e[k] = 1.0
                out[i][:, o[k]] = di[i] * np.interp(di[i], xs, e) / xs[k]
    return out


H = anchor_dollars(dur)


# ----------------------------------------------------------------------------- weights
def carried_weights(sig: pd.DataFrame, cols: list[str], grp: pd.Series | None = None, min_obs: int = 5):
    """Zero-sum rank weights (within bucket if `grp` is given), sum |w| = 1, with the corrected carry-forward.
    Returns (weights, first date with a book)."""
    s = sig[cols].ffill(limit=CARRY_DAYS)
    if grp is not None:
        s = stats.demean_by_group(s, grp[cols])
    r = stats.demean_by_group(s.rank(axis=1), None if grp is None else grp[cols])
    w = r.div(r.abs().sum(axis=1).where(s.notna().sum(axis=1) >= min_obs), axis=0)
    first = w.dropna(how="all").index.min()
    w = w.where(w.notna().any(axis=1)).ffill()  # too few funds: keep the whole previous book
    return w.fillna(0.0), first


def per_gross(w: pd.DataFrame) -> pd.DataFrame:
    g = w.abs().sum(axis=1)
    return w.div(g.where(g > 0), axis=0).fillna(0.0)


def held(w0: pd.DataFrame, h: int) -> pd.DataFrame:
    """h-day holding period: average of the last h days' weight vectors, rescaled to $1 of gross."""
    return per_gross(w0.rolling(h, min_periods=1).mean() if h > 1 else w0)


# ----------------------------------------------------------------------------- paper book
def evaluate(w: pd.DataFrame, ret_h: pd.DataFrame, Hm: np.ndarray, first) -> tuple[dict, pd.DataFrame]:
    """Paper PnL of dollar weights `w` (decided at T, per $1 of fund gross) on the hedged returns `ret_h`.

    Costs: $0.0035/share on fund legs and on the hedge legs implied by `Hm`. Financing: bill rate on the net
    dollars of funds plus hedge legs. No volume cap. Returns (summary, daily frame)."""
    cols = list(w.columns)
    w = w.copy()
    w.loc[w.index < first] = 0.0
    pnl = (w.shift(1) * ret_h[cols]).sum(axis=1)
    fund_cost = ((w - w.shift(1)).abs() * COST / p.close[cols]).sum(axis=1)
    Hn = np.nan_to_num(Hm[:, [tick_idx[c] for c in cols], :])
    Hnext = np.vstack([Hn[1:], Hn[-1:]])  # the hedge of a book decided at T is sized for the return of T+1
    hp = pd.DataFrame(-(w.to_numpy()[:, :, None] * Hnext).sum(axis=1), index=w.index, columns=anch)
    hedge_cost = ((hp - hp.shift(1)).abs() * COST / p.close[anch]).sum(axis=1)
    net = (w.sum(axis=1) + hp.sum(axis=1)).shift(1)
    fin = net * bill.reindex(w.index).ffill() / 252
    d = pd.DataFrame({"pnl": pnl, "fund_cost": fund_cost, "hedge_cost": hedge_cost, "fin": fin}).loc[first:].iloc[1:].fillna(0.0)
    d["net"] = d.pnl - d.fin - d.fund_cost - d.hedge_cost
    m0304 = (d.index.year == 2020) & (d.index.month.isin([3, 4]))
    top = d.pnl.abs().nlargest(10).index
    cum = d.net.cumsum()
    summary = {
        "ann_ret_pct": d.pnl.mean() * 252 * 100,
        "vol_pct": d.pnl.std() * np.sqrt(252) * 100,
        "sharpe_project": stats.sharpe(d.pnl),
        "sharpe_net_fin": stats.sharpe(d.pnl - d.fin),
        "sharpe_after_cost": stats.sharpe(d.net),
        "t_after_cost": d.net.mean() / d.net.std() * np.sqrt(len(d)),
        "cost_funds_pct": d.fund_cost.mean() * 252 * 100,
        "cost_hedge_pct": d.hedge_cost.mean() * 252 * 100,
        "turnover_day": (w - w.shift(1)).abs().sum(axis=1).loc[first:].mean(),
        "after_cost_ex2020": stats.sharpe(d.net[d.index.year > 2020]),
        "after_cost_ex_MarApr20": stats.sharpe(d.net[~m0304]),
        "top10_share_of_pnl": d.pnl[top].sum() / d.pnl.sum() if d.pnl.sum() else np.nan,
        "maxdd_pct": (cum.cummax() - cum).max() * 100,
        "worst_day_bp": d.pnl.min() * 1e4,
    }
    return summary, d


def by_year(d: pd.DataFrame, col: str = "net") -> dict:
    return d[col].groupby(d.index.year).apply(stats.sharpe).round(2).to_dict()


# ----------------------------------------------------------------------------- engine
class Lookup(WeightStrategy):
    """Hands precomputed weights to the engine. Row T must be built from information available at the day-T
    decision; the engine then supplies fills at the close under the 3% cap, costs, corporate actions and PnL."""

    def __init__(self, name: str, weights: pd.DataFrame):
        self.name, self.W = name, weights

    def target_weights(self, ctx):
        return self.W.loc[ctx.date] if ctx.date in self.W.index else pd.Series(dtype=float)


def with_hedge(w: pd.DataFrame, Hm: np.ndarray) -> pd.DataFrame:
    """Fund weights plus the Treasury hedge legs; row T of `Hm` holds hedge dollars from durations through T-1."""
    Hn = np.nan_to_num(Hm[:, [tick_idx[c] for c in w.columns], :])
    hp = pd.DataFrame(-(w.to_numpy()[:, :, None] * Hn).sum(axis=1), index=w.index, columns=anch)
    return pd.concat([w, hp], axis=1)


def run_engine(name: str, w_full: pd.DataFrame, scale: float):
    """Run ticker-labelled weights (fractions of the $57MM gross target after `scale`) through the Backtester.
    Returns (metrics, Sharpe by year net of financing, BacktestResult)."""
    W = (w_full * scale).rename(columns=sym)
    res = Backtester(market, Lookup(name, W), BacktestConfig(), record_orders=False, record_targets=True).run()
    d = res.daily
    m = res.metrics(None, bill)
    r = (d["pnl"] - financing_charge(d["long_mv"] + d["short_mv"], bill, 252)) / 1e7
    close = market.eod["close"]
    tgt = res.targets.reindex(columns=market.symbols).fillna(0.0)
    miss = ((res.positions - tgt).abs() * close).sum(axis=1) / ((tgt.abs() * close).sum(axis=1)).replace(0, np.nan)
    active = d["gmv"] > 0
    m0304 = (d.index.year == 2020) & (d.index.month.isin([3, 4]))
    out = {
        "avg_gmv_mm": d["gmv"].mean() / 1e6,
        "max_gmv_mm": d["gmv"].max() / 1e6,
        "ann_pnl_k": m["annualized_net_pnl"] / 1e3,
        "ann_costs_k": m["annualized_costs"] / 1e3,
        "ann_vol_pct": m["annualized_vol"] * 100,
        "sharpe_project": m["sharpe"],
        "sharpe_net_fin": m["sharpe_net_of_financing"],
        "gross_sharpe_project": m["gross_sharpe"],
        "t_net_fin": m["sharpe_net_of_financing"] * np.sqrt(len(d) / 252),
        "net_fin_ex2020": stats.sharpe(r[d.index.year > 2020]),
        "net_fin_ex_MarApr20": stats.sharpe(r[~m0304]),
        "maxdd_pct": m["max_drawdown_pct"] * 100,
        "maxdd_usd_k": m["max_drawdown_usd"] / 1e3,
        "worst_day_k": d["pnl"].min() / 1e3,
        "avg_net_expo_mm": m["avg_net_exposure"] / 1e6,
        "avg_dv01": m["avg_dv01"],
        "max_abs_dv01": m["max_abs_dv01"],
        "breach_days": m["gmv_breach_days"],
        "median_unfilled": miss[active].median(),
        "p90_unfilled": miss[active].quantile(0.9),
    }
    return out, r.groupby(d.index.year).apply(stats.sharpe).round(2).to_dict(), res


def merton_tier_signals():
    """Tier model in leverage space and the naive spread-gap benchmark, as in the research notebook."""

    def credit_vol(t, tenor="vol3M"):
        rates_only = merton.rates_vol(dur[t], p.vol[tenor][anch], dur[anch])
        rho = (ylds[t] - gs[t] / 100).diff().rolling(250, min_periods=60).corr(gs[t].diff()).shift(1).fillna(-0.2)
        return merton.credit_only_vol(p.vol[tenor][t], rates_only, rho) / 100

    sol = merton.solve_panel(gs["HYG"] / 1e4, credit_vol("HYG"), ttm["HYG"], (ylds["HYG"] - gs["HYG"] / 100) / 100)
    tier = ["JNK", "USHY", "SPHY", "SHYG", "SJNK", "BKLN", "SRLN"]
    sig_a = sol.sigma_a.rolling(20, min_periods=5).median()
    log_lev = pd.DataFrame({t: np.log(merton.implied_leverage_panel(gs[t] / 1e4, sig_a, ttm[t], (ylds[t] - gs[t] / 100) / 100)) for t in tier})
    lev = stats.trailing_z(log_lev, 60, 20).shift(1)
    naive = stats.trailing_z(gs[tier].sub(gs["HYG"], axis=0), 60, 20).shift(1)
    return tier, lev, naive


def pair_position(leg: str) -> pd.Series:
    """Position in (leg - HYG), known at T: the trailing-120d z of the log implied-hazard gap, clipped at +/-3
    and divided by 3, carried forward when missing."""
    gap = hazard.log_hazard_gap(gs[leg], gs["HYG"], RECOVERY[groups[leg]], RECOVERY["HY"])
    return stats.trailing_z(gap, 120, 40).shift(1).ffill(limit=CARRY_DAYS).clip(-3, 3) / 3.0
