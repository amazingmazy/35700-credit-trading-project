"""Load, clean and align the supplied ETF data into date x symbol panels.

Data-preparation decisions (also documented in README):
  * Universe = ISINs in etf_symbology; every other instrument is dropped.
  * Trading calendar = dates present in BOTH the EOD and 3PM files. The EOD file has
    extra dates (Columbus/Veterans Day bond holidays) with no 3PM snapshot; they are removed.
  * Prices are raw (unadjusted). Splits are applied to positions, and cash
    distributions are paid to holders, on ex_date_adj (mapped to the next trading date
    if it falls on a removed holiday).
  * Corporate action types map to CASH (paid per share), SPLIT (share ratio) or IGNORED.
  * Risk fields (duration, dv01) with impossible values are set to NaN.
  * 3PM decision price = 3PM `last`, falling back to the 3PM bid/ask mid.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EOD_FIELDS = [
    "open", "high", "low", "close", "bid", "ask", "vwap", "volume", "vwapvol", "vwaptrd",
    "shs", "mktcap", "fnav", "fta", "fflow", "duration", "dv01", "yield", "spread",
    "ispread", "gspread", "avgttm", "avgcpn", "vol1M", "vol3M", "vol6M", "vol12M",
    "callvol", "putvol", "opintcall", "opintput",
]
SNAP_FIELDS = ["last", "bid", "ask", "mid", "price"]

CASH_TYPES = frozenset({
    "Income", "Regular Cash", "Special Cash", "Long Term Cap Gain", "Short Term Cap Gain",
    "Capital Gains", "Return of Capital", "Partnership Dist", "Interim", "Daily Accrual",
})
SPLIT_TYPES = frozenset({"Stock Split"})
# Everything else (Omitted, Cancelled, Discontinued, ...) is ignored.

MAX_VALID_DURATION = 50.0


def classify_corp_action(dividend_type: str) -> str:
    if dividend_type in CASH_TYPES:
        return "cash"
    if dividend_type in SPLIT_TYPES:
        return "split"
    return "ignored"


@dataclass
class MarketData:
    """Aligned, read-only market data.

    eod[field]  : DataFrame (dates x symbols), EOD values observed at the close of each date.
    snap[field] : DataFrame (dates x symbols), 3PM snapshot (last, bid, ask, mid, price).
    corp_actions: long DataFrame of events with `kind` and `trade_date` (effective trading date).
    split_ratio : DataFrame (dates x symbols), share multiplier effective at the start of each date (1 = none).
    cash_dividend: DataFrame (dates x symbols), cash per share paid to holders at the prior close.
    total_return: DataFrame (dates x symbols), close-to-close total return incl. splits & dividends.
    """

    dates: pd.DatetimeIndex
    symbols: pd.Index
    eod: dict[str, pd.DataFrame]
    snap: dict[str, pd.DataFrame]
    corp_actions: pd.DataFrame
    split_ratio: pd.DataFrame
    cash_dividend: pd.DataFrame
    total_return: pd.DataFrame
    symbology: pd.DataFrame = field(default_factory=pd.DataFrame)

    # ------------------------------------------------------------------ build
    @classmethod
    def from_frames(
        cls,
        eod: pd.DataFrame,
        snap: pd.DataFrame,
        corp_actions: pd.DataFrame,
        symbology: pd.DataFrame,
    ) -> "MarketData":
        universe = pd.Index(symbology["symbol"].unique())
        eod = eod[eod["symbol"].isin(universe)].drop_duplicates(["symbol", "date"], keep="last")
        snap = snap[snap["symbol"].isin(universe)].drop_duplicates(["symbol", "date"], keep="last")

        eod_dates = pd.DatetimeIndex(eod["date"].unique())
        snap_dates = pd.DatetimeIndex(snap["date"].unique())
        dates = eod_dates.intersection(snap_dates).sort_values()
        dropped = eod_dates.difference(snap_dates).union(snap_dates.difference(eod_dates))
        if len(dropped):
            logger.info("Dropping %d non-common calendar dates: %s", len(dropped), list(dropped.date))
        symbols = pd.Index(sorted(set(eod["symbol"]) & set(universe)), name="symbol")

        def panels(df: pd.DataFrame, cols: list[str]) -> dict[str, pd.DataFrame]:
            """Scatter long rows into (dates x symbols) arrays via integer positions (fast pivot)."""
            ri = dates.get_indexer(df["date"])
            ci = symbols.get_indexer(df["symbol"])
            keep = (ri >= 0) & (ci >= 0)
            ri, ci = ri[keep], ci[keep]
            out = {}
            for col in cols:
                arr = np.full((len(dates), len(symbols)), np.nan)
                arr[ri, ci] = df[col].to_numpy(dtype=float)[keep]
                out[col] = pd.DataFrame(arr, index=dates, columns=symbols)
            return out

        eod_panels = panels(eod, [f for f in EOD_FIELDS if f in eod.columns])
        _clean_risk_fields(eod_panels)

        snap_panels = panels(snap, [f for f in ("last", "bid", "ask") if f in snap.columns])
        bid, ask = snap_panels.get("bid"), snap_panels.get("ask")
        mid = (bid + ask) / 2 if bid is not None and ask is not None else None
        if mid is not None:
            mid = mid.where((bid > 0) & (ask > 0) & (ask >= bid))
            snap_panels["mid"] = mid
        last = snap_panels.get("last")
        price = last.where(last > 0) if last is not None else mid
        if mid is not None:
            price = price.fillna(mid)
        snap_panels["price"] = price

        ca, split_ratio, cash_div = _build_corp_actions(corp_actions, dates, symbols)
        close = eod_panels["close"]
        prev_close = close.shift(1)
        total_return = (split_ratio * (close + cash_div)) / prev_close - 1.0

        sym = symbology[symbology["symbol"].isin(symbols)].reset_index(drop=True)
        return cls(dates, symbols, eod_panels, snap_panels, ca, split_ratio, cash_div, total_return, sym)

    # ------------------------------------------------------------- helpers
    def history_count(self, t_index: int) -> np.ndarray:
        """Number of valid closes per symbol over dates strictly before dates[t_index]."""
        cache = getattr(self, "_hist_count", None)
        if cache is None:
            c = self.eod["close"].notna().to_numpy().cumsum(axis=0)
            cache = np.vstack([np.zeros((1, c.shape[1]), dtype=int), c])
            self._hist_count = cache
        return cache[t_index]

    def symbol_for(self, ticker: str) -> str:
        """Map a ticker (e.g. 'AGG') to its ISIN symbol."""
        hit = self.symbology.loc[self.symbology["ticker"] == ticker, "symbol"]
        if hit.empty:
            raise KeyError(f"Unknown ticker {ticker!r}")
        return hit.iloc[0]

    def ticker_for(self, symbol: str) -> str:
        hit = self.symbology.loc[self.symbology["symbol"] == symbol, "ticker"]
        return hit.iloc[0] if not hit.empty else symbol

    def tickers(self) -> pd.Series:
        """Series symbol -> ticker for all symbols."""
        m = dict(zip(self.symbology.get("symbol", []), self.symbology.get("ticker", [])))
        return pd.Series([m.get(s, s) for s in self.symbols], index=self.symbols)

    def subset(self, start=None, end=None) -> "MarketData":
        """Restrict to a date range (inclusive)."""
        mask = np.ones(len(self.dates), dtype=bool)
        if start is not None:
            mask &= self.dates >= pd.Timestamp(start)
        if end is not None:
            mask &= self.dates <= pd.Timestamp(end)
        d = self.dates[mask]
        ca = self.corp_actions
        ca = ca[ca["trade_date"].isin(d)]
        return MarketData(
            d, self.symbols,
            {k: v.loc[d] for k, v in self.eod.items()},
            {k: v.loc[d] for k, v in self.snap.items()},
            ca, self.split_ratio.loc[d], self.cash_dividend.loc[d], self.total_return.loc[d],
            self.symbology,
        )


def _clean_risk_fields(p: dict[str, pd.DataFrame]) -> None:
    if "duration" in p:
        bad = (p["duration"] < 0) | (p["duration"] > MAX_VALID_DURATION)
        p["duration"] = p["duration"].mask(bad)
        if "dv01" in p:
            p["dv01"] = p["dv01"].mask(bad)
    if "dv01" in p:
        p["dv01"] = p["dv01"].mask(p["dv01"] < 0)
        if "close" in p:
            # dv01 per 100bp cannot exceed the price by a wide margin for these funds
            p["dv01"] = p["dv01"].mask(p["dv01"] > p["close"] * MAX_VALID_DURATION / 100)


def _build_corp_actions(ca: pd.DataFrame, dates: pd.DatetimeIndex, symbols: pd.Index):
    ca = ca[ca["symbol"].isin(symbols)].copy()
    ca = ca.drop_duplicates(["symbol", "ex_date_adj", "dividend_type", "dividend_amount"])
    ca["kind"] = ca["dividend_type"].map(classify_corp_action)
    ca["dividend_amount"] = ca["dividend_amount"].astype(float)

    # Effective trading date: first calendar trading date >= ex_date_adj.
    ex = pd.to_datetime(ca["ex_date_adj"]).to_numpy()
    pos = dates.searchsorted(ex, side="left")
    in_range = pos < len(dates)
    ca["trade_date"] = pd.NaT
    ca.loc[in_range, "trade_date"] = dates[pos[in_range]]
    # Events before the first date or after the last date have no effect on the backtest.
    ca.loc[pd.to_datetime(ca["ex_date_adj"]) < dates[0], "trade_date"] = pd.NaT
    ca["declared_date"] = pd.to_datetime(ca["declared_date"])
    ca = ca.sort_values(["trade_date", "symbol"]).reset_index(drop=True)

    split_ratio = pd.DataFrame(1.0, index=dates, columns=symbols)
    cash = pd.DataFrame(0.0, index=dates, columns=symbols)
    eff = ca.dropna(subset=["trade_date"])
    for row in eff[eff["kind"] == "split"].itertuples():
        if row.dividend_amount and row.dividend_amount > 0:
            split_ratio.loc[row.trade_date, row.symbol] *= row.dividend_amount
    cash_rows = eff[(eff["kind"] == "cash")].dropna(subset=["dividend_amount"])
    if not cash_rows.empty:
        agg = cash_rows.groupby(["trade_date", "symbol"])["dividend_amount"].sum().unstack()
        cash = cash.add(agg.reindex(index=dates, columns=symbols).fillna(0.0))
    return ca, split_ratio, cash


class DataLoader:
    """Reads the four supplied parquet files and builds a MarketData object."""

    FILES = {
        "symbology": "etf_symbology.parquet",
        "eod": "etf_data_eod.parquet",
        "snap": "etf_data_3pm.parquet",
        "corp_actions": "etf_dividends_splits.parquet",
    }

    def __init__(self, data_dir: str | Path = "data"):
        self.data_dir = Path(data_dir)

    def read_raw(self) -> dict[str, pd.DataFrame]:
        out = {}
        for key, fname in self.FILES.items():
            path = self.data_dir / fname
            if not path.exists():
                raise FileNotFoundError(path)
            out[key] = pd.read_parquet(path)
        return out

    def load(self, start=None, end=None) -> MarketData:
        raw = self.read_raw()
        md = MarketData.from_frames(raw["eod"], raw["snap"], raw["corp_actions"], raw["symbology"])
        if start is not None or end is not None:
            md = md.subset(start, end)
        logger.info("Loaded %d dates x %d symbols", len(md.dates), len(md.symbols))
        return md
