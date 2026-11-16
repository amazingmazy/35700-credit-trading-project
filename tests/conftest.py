"""Synthetic market fixtures: 3 universe ETFs (A, B, C) + 1 non-universe instrument (X),
13 business days of which one (HOLIDAY) exists only in the EOD file."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import BacktestConfig
from src.data_loader import MarketData
from src.strategies.base import Strategy

SYMBOLS = ["A", "B", "C"]
TICKERS = {"A": "AAA", "B": "BBB", "C": "CCC"}
ALL_DATES = pd.bdate_range("2024-01-02", periods=13)
HOLIDAY = ALL_DATES[6]
TRADING_DATES = ALL_DATES.drop(HOLIDAY)


def base_prices() -> dict[str, np.ndarray]:
    n = len(ALL_DATES)
    i = np.arange(n, dtype=float)
    return {
        "A": 100.0 + i,  # rising 1/day
        "B": np.full(n, 50.0),  # flat
        "C": 20.0 - 0.5 * i,  # falling
        "X": np.full(n, 10.0),  # not in universe
    }


def make_frames(prices=None, volume=1_000_000.0, corp_rows=None, snap_last_nan=()):
    """Long-format EOD / 3PM / corp-action / symbology frames like the supplied parquet files.

    snap_last_nan: iterable of (symbol, date) pairs whose 3PM `last` is missing.
    volume: scalar or dict symbol -> scalar/array.
    """
    prices = prices or base_prices()
    eod_rows, snap_rows = [], []
    for sym, px in prices.items():
        vol = volume.get(sym, 1_000_000.0) if isinstance(volume, dict) else volume
        vol = np.broadcast_to(np.asarray(vol, dtype=float), px.shape)
        for k, d in enumerate(ALL_DATES):
            c = px[k]
            eod_rows.append({
                "symbol": sym, "date": d, "open": c, "high": c, "low": c, "close": c,
                "bid": c - 0.01, "ask": c + 0.01, "volume": vol[k],
                "duration": 5.0, "dv01": c * 5.0 / 100.0, "spread": 100.0,
            })
            if d != HOLIDAY:
                last = np.nan if (sym, d) in set(snap_last_nan) else c - 0.05
                snap_rows.append({"symbol": sym, "date": d, "last": last, "bid": c - 0.06, "ask": c - 0.04})
    corp_cols = ["symbol", "declared_date", "ex_date", "record_date", "payable_date",
                 "dividend_amount", "dividend_frequency", "dividend_type", "ex_date_adj"]
    corp = pd.DataFrame(corp_rows or [], columns=corp_cols)
    for c in ("declared_date", "ex_date", "record_date", "payable_date", "ex_date_adj"):
        corp[c] = pd.to_datetime(corp[c])
    symbology = pd.DataFrame({"symbol": SYMBOLS, "ticker": [TICKERS[s] for s in SYMBOLS]})
    return pd.DataFrame(eod_rows), pd.DataFrame(snap_rows), corp, symbology


def corp_event(symbol, ex_date, amount, dtype="Income", declared=None):
    ex = pd.Timestamp(ex_date)
    return {
        "symbol": symbol, "declared_date": pd.Timestamp(declared) if declared else ex - pd.Timedelta(days=3),
        "ex_date": ex, "record_date": ex, "payable_date": ex + pd.Timedelta(days=3),
        "dividend_amount": amount, "dividend_frequency": "Monthly", "dividend_type": dtype, "ex_date_adj": ex,
    }


def vectorized_price_plus_dividend_pnl(market: MarketData, positions: pd.DataFrame) -> pd.Series:
    """Independent check of the ledger: $ held at the prior close x split/dividend-adjusted total return."""
    close = market.eod["close"].ffill()
    tr = (market.split_ratio * (close + market.cash_dividend)) / close.shift(1) - 1
    held_value = positions.shift(1) * close.shift(1)
    return (held_value * tr).where(held_value != 0, 0.0).sum(axis=1)


def make_market(**kwargs) -> MarketData:
    return MarketData.from_frames(*make_frames(**kwargs))


class FixedTargets(Strategy):
    """Holds a fixed share target. hold_after_first=True returns NaN after day 0 (keep positions)."""

    name = "fixed_targets"

    def __init__(self, targets: dict, hold_after_first: bool = False):
        self.targets = targets
        self.hold_after_first = hold_after_first
        self.calls = 0

    def generate_targets(self, ctx):
        self.calls += 1
        if self.hold_after_first and self.calls > 1:
            return pd.Series(np.nan, index=ctx.symbols)
        return pd.Series(self.targets, dtype=float)


@pytest.fixture
def market() -> MarketData:
    return make_market()


@pytest.fixture
def config() -> BacktestConfig:
    return BacktestConfig()
