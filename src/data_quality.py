"""Data-quality checks on the supplied corporate-action records, and analytics cleaning helpers.

Why this module exists (found on the supplied files, 2026-11-16):
  * The dividend file carries no cash-distribution record for most ETFs between Dec 2024 and Jul 2025
    (about 41 funds have a record in each of Feb-Jul 2025 against about 205 in a normal month).
    Prices still drop on the ex-date, so a held fund books a false loss of about one month of income.
  * Two reverse splits are not in the file (the close doubles overnight with no split record), so a
    held fund books a false gain of about 100%.
The engine books PnL from the recorded events only, so these gaps go straight into results.

Group decision: ETFs with either problem are excluded from trading for the WHOLE sample
(`BacktestConfig.clean_universe_only`, default True); the rest are handled later in a separate,
documented imputation track.

NOTE on point-in-time: the exclusion list is derived from the full sample. It is a data-integrity
filter on the supplied corporate-action records, not a selection on performance. It is stated as an
assumption in the README and DEVIATIONS.md.

Rules (all data-driven, no hard-coded dates or tickers):
  gap month        : fewer than CASH_GAP_RATIO times as many ETFs have a cash record as in the same
                     month one year earlier, and the shortfall is at least MIN_SHORTFALL_FRACTION of the
                     universe. The final month of the sample is skipped (may be partial).
  missing dividend : the ETF had a cash record in the same month one year earlier and has none in a gap month.
  unrecorded split : close_t / close_{t-1} is within SPLIT_TOLERANCE of k or 1/k for k in SPLIT_FACTORS
                     and no split is recorded for that date.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CASH_GAP_RATIO = 0.90
MIN_SHORTFALL_FRACTION = 0.10
SPLIT_FACTORS = (2, 3, 4, 5, 10)
SPLIT_TOLERANCE = 0.03

# Analytics cleaning (yield in %, spreads in bp, duration in years)
YIELD_BOUNDS = (-5.0, 40.0)
DURATION_BOUNDS = (0.0, 40.0)
MAX_ABS_SPREAD_BP = 3000.0
ANALYTIC_FIELDS = ("duration", "dv01", "yield", "spread", "ispread", "gspread", "avgttm", "avgcpn")
SPREAD_FIELDS = ("spread", "ispread", "gspread")


# --------------------------------------------------------------------------- dividends
def monthly_payers(cash_dividend: pd.DataFrame) -> pd.DataFrame:
    """Boolean (month x symbol): at least one cash distribution went ex in that month.

    The index is a complete monthly PeriodIndex from the first to the last date of the panel.
    """
    paid = cash_dividend.ne(0.0) & cash_dividend.notna()
    by_month = paid.groupby(cash_dividend.index.to_period("M")).any()
    months = pd.period_range(by_month.index.min(), by_month.index.max(), freq="M")
    return by_month.reindex(months, fill_value=False)


def gap_months(
    payers: pd.DataFrame,
    ratio: float = CASH_GAP_RATIO,
    min_shortfall_fraction: float = MIN_SHORTFALL_FRACTION,
) -> pd.PeriodIndex:
    """Months in which the file as a whole is short of cash records relative to a year earlier.

    Both conditions must hold: the count of paying ETFs is below `ratio` times the count a year earlier,
    and the shortfall is at least `min_shortfall_fraction` of the universe. The second condition keeps
    thin months (e.g. January, when few funds go ex) from being flagged when a handful of low-yield
    funds legitimately pay nothing.
    """
    n = payers.sum(axis=1)
    prev = n.shift(12)
    is_gap = prev.gt(0) & n.lt(ratio * prev) & (prev - n).ge(min_shortfall_fraction * payers.shape[1])
    if len(is_gap):
        is_gap.iloc[-1] = False  # the last month of the sample may be partial
    return payers.index[is_gap.to_numpy()]


def missing_dividend_months(payers: pd.DataFrame, gaps: pd.PeriodIndex) -> pd.DataFrame:
    """Boolean (gap month x symbol): paid in the same month a year earlier, no record in the gap month."""
    paid_year_earlier = payers.shift(12, fill_value=False)
    return (paid_year_earlier & ~payers).loc[gaps]


# --------------------------------------------------------------------------- splits
def unrecorded_splits(
    close: pd.DataFrame,
    split_ratio: pd.DataFrame,
    factors: tuple[int, ...] = SPLIT_FACTORS,
    tolerance: float = SPLIT_TOLERANCE,
) -> pd.DataFrame:
    """Long frame (date, symbol, price_ratio) of split-sized overnight price jumps with no split record."""
    ratio = close / close.ffill().shift(1)
    hit = pd.DataFrame(False, index=close.index, columns=close.columns)
    for k in factors:
        hit |= (ratio * k - 1.0).abs().lt(tolerance) | (ratio / k - 1.0).abs().lt(tolerance)
    hit &= split_ratio.reindex_like(close).eq(1.0)
    rows, cols = np.nonzero(hit.to_numpy())
    return pd.DataFrame({
        "date": close.index[rows],
        "symbol": close.columns[cols],
        "price_ratio": ratio.to_numpy()[rows, cols],
    })


# --------------------------------------------------------------------------- report
@dataclass(frozen=True)
class DataQualityReport:
    symbols: pd.Index
    gap_months: pd.PeriodIndex
    missing_dividends: pd.DataFrame  # bool, gap months x symbols
    unrecorded_splits: pd.DataFrame  # long: date, symbol, price_ratio
    excluded: pd.Index = field(default_factory=lambda: pd.Index([]))

    @property
    def clean(self) -> pd.Index:
        return self.symbols.difference(self.excluded, sort=False)

    @property
    def excluded_mask(self) -> np.ndarray:
        return self.symbols.isin(self.excluded)

    def summary(self) -> dict:
        n_missing = self.missing_dividends.sum(axis=0)
        return {
            "n_symbols": int(len(self.symbols)),
            "n_clean": int(len(self.clean)),
            "n_excluded": int(len(self.excluded)),
            "gap_months": [str(m) for m in self.gap_months],
            "n_symbols_missing_dividends": int((n_missing > 0).sum()),
            "missing_by_month": {str(m): int(v) for m, v in self.missing_dividends.sum(axis=1).items()},
            "unrecorded_splits": [
                {"date": str(pd.Timestamp(r.date).date()), "symbol": r.symbol, "price_ratio": round(float(r.price_ratio), 4)}
                for r in self.unrecorded_splits.itertuples()
            ],
        }


def assess(cash_dividend: pd.DataFrame, close: pd.DataFrame, split_ratio: pd.DataFrame) -> DataQualityReport:
    """Run both checks and build the exclusion list (symbols with a missing dividend or an unrecorded split)."""
    symbols = close.columns
    payers = monthly_payers(cash_dividend)
    gaps = gap_months(payers)
    missing = missing_dividend_months(payers, gaps)
    splits = unrecorded_splits(close, split_ratio)
    bad = missing.any(axis=0).reindex(symbols, fill_value=False).to_numpy() | symbols.isin(splits["symbol"])
    return DataQualityReport(symbols, gaps, missing, splits, symbols[bad])


def reconcile_distributions(cash_dividend: pd.DataFrame, close: pd.DataFrame, yield_pct: pd.DataFrame) -> pd.DataFrame:
    """Per symbol and calendar year: recorded distributions / average close (%) next to the median yield field (%).

    Long frame with columns symbol, year, dist_yield_pct, median_yield_pct, shortfall_pct.
    """
    year = cash_dividend.index.year
    dist = cash_dividend.groupby(year).sum() / close.groupby(year).mean() * 100.0
    med = yield_pct.groupby(year).median()
    out = pd.DataFrame({
        "dist_yield_pct": dist.stack(future_stack=True),
        "median_yield_pct": med.stack(future_stack=True),
    })
    out["shortfall_pct"] = out["median_yield_pct"] - out["dist_yield_pct"]
    out.index.names = ["year", "symbol"]
    return out.reset_index()[["symbol", "year", "dist_yield_pct", "median_yield_pct", "shortfall_pct"]]


# --------------------------------------------------------------------------- analytics cleaning
def clean_analytics(eod: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Return cleaned copies of the analytics panels.

    A (date, symbol) row is treated as bad if its yield or duration is outside the plausible range or any
    spread field exceeds MAX_ABS_SPREAD_BP in absolute value; every analytics field is set to NaN on bad rows.
    Rows with a missing yield or duration are left as they are.
    """
    present = [f for f in ANALYTIC_FIELDS if f in eod]
    if not present:
        return {}
    ref = eod[present[0]]
    bad = pd.DataFrame(False, index=ref.index, columns=ref.columns)
    if "yield" in eod:
        y = eod["yield"]
        bad |= y.lt(YIELD_BOUNDS[0]) | y.gt(YIELD_BOUNDS[1])
    if "duration" in eod:
        d = eod["duration"]
        bad |= d.lt(DURATION_BOUNDS[0]) | d.gt(DURATION_BOUNDS[1])
    for f in SPREAD_FIELDS:
        if f in eod:
            bad |= eod[f].abs().gt(MAX_ABS_SPREAD_BP)
    return {f: eod[f].mask(bad) for f in present}


def despike(
    panel: pd.DataFrame,
    window: int = 21,
    mad_window: int = 63,
    n_mad: float = 8.0,
    rel: float = 0.35,
    floor: float = 15.0,
) -> pd.DataFrame:
    """Set isolated spikes to NaN using only past data.

    A value is a spike if it differs from the trailing `window`-day median (ending the day before) by more
    than max(n_mad x trailing median absolute deviation, rel x |median|, floor). Units follow the panel
    (the defaults suit spreads in bp).
    """
    med = panel.rolling(window, min_periods=5).median().shift(1)
    dev = (panel - med).abs()
    mad = dev.rolling(mad_window, min_periods=10).median().shift(1)
    threshold = np.maximum(np.maximum(n_mad * mad, rel * med.abs()), floor)
    return panel.mask(dev > threshold)
