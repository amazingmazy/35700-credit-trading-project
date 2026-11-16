"""Ticker-labelled research panels built from MarketData, and the duration-matched rates hedge."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data_loader import MarketData
from src.data_quality import SPREAD_FIELDS, clean_analytics, despike
from src.research.universe import HEDGE_ANCHORS, bucket_of

VOL_TENORS = ("vol1M", "vol3M", "vol6M", "vol12M")


@dataclass
class Panels:
    """Date x ticker panels. Values dated T are observed at the close of T unless stated otherwise;
    use `.shift(1)` (or `lagged()`) before feeding them to a day-T decision."""

    tickers: list[str]
    bucket: pd.Series  # ticker -> bucket
    close: pd.DataFrame
    nav: pd.DataFrame
    volume: pd.DataFrame
    price_3pm: pd.DataFrame  # 3PM last, else mid (the engine's decision price)
    mid_3pm: pd.DataFrame
    ret: pd.DataFrame  # close(T-1) -> close(T) total return
    ret_to_3pm: pd.DataFrame  # close(T-1) -> 3PM(T), using price_3pm
    ret_to_3pm_mid: pd.DataFrame  # close(T-1) -> 3PM(T), using mid_3pm
    ret_3pm_to_close: pd.DataFrame  # 3PM(T) -> close(T), from price_3pm
    analytics: dict[str, pd.DataFrame] = field(default_factory=dict)  # cleaned duration, yield, spreads, ...
    vol: dict[str, pd.DataFrame] = field(default_factory=dict)  # option implied vol by tenor (%)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.close.index

    def a(self, name: str) -> pd.DataFrame:
        return self.analytics[name]

    def cap_usd(self, participation: float = 0.03, window: int = 60) -> pd.DataFrame:
        """Dollar size of the volume cap known at the day-T decision: participation x trailing-median
        dollar volume through T-1."""
        return participation * (self.volume * self.close).rolling(window, min_periods=20).median().shift(1)


def build_panels(market: MarketData, clean_only: bool = True, despike_spreads: bool = True) -> Panels:
    """Research panels for the clean ETFs (or all ETFs), with columns labelled by ticker."""
    keep = ~market.excluded_mask() if clean_only else np.ones(len(market.symbols), dtype=bool)
    symbols = market.symbols[keep]
    tick = market.tickers()[symbols]

    def t(panel: pd.DataFrame) -> pd.DataFrame:
        out = panel[symbols].copy()
        out.columns = pd.Index(tick.to_numpy(), name="ticker")
        return out

    close, split, cash = t(market.eod["close"]), t(market.split_ratio), t(market.cash_dividend)
    prev_close = close.shift(1)
    price, mid = t(market.snap["price"]), t(market.snap["mid"])
    analytics = {k: t(v) for k, v in clean_analytics(market.eod).items()}
    if despike_spreads:
        for f in SPREAD_FIELDS:
            if f in analytics:
                analytics[f] = despike(analytics[f])
    return Panels(
        tickers=list(tick),
        bucket=bucket_of(tick.to_numpy()),
        close=close,
        nav=t(market.eod["fnav"]) if "fnav" in market.eod else close * np.nan,
        volume=t(market.eod["volume"]),
        price_3pm=price,
        mid_3pm=mid,
        ret=t(market.total_return),
        ret_to_3pm=(split * price + cash) / prev_close - 1.0,
        ret_to_3pm_mid=(split * mid + cash) / prev_close - 1.0,
        ret_3pm_to_close=close / price - 1.0,
        analytics=analytics,
        vol={k: t(market.eod[k]) for k in VOL_TENORS if k in market.eod},
    )


def rates_return(ret: pd.DataFrame, duration: pd.DataFrame, anchors=HEDGE_ANCHORS) -> pd.DataFrame:
    """Return of a duration-matched Treasury position for each column of `ret`.

    Each day the anchors' returns per year of duration, k_a = ret_a / D_a, are interpolated linearly in
    duration (flat beyond the shortest and longest anchor) and multiplied by the column's own duration.
    Durations are those known the day before (forward-filled, lagged one row), so the hedge ratio is
    available at decision time. `ret` can be a close-to-close, close-to-3PM or 3PM-to-close return
    panel; the anchors' returns are taken from the same panel.
    """
    dur = duration.reindex(columns=ret.columns).ffill().shift(1)
    k = (ret[list(anchors)] / dur[list(anchors)]).to_numpy()
    xa, di = dur[list(anchors)].to_numpy(), dur.to_numpy()
    out = np.full(di.shape, np.nan)
    for i in range(len(ret)):
        if np.isfinite(k[i]).all() and np.isfinite(xa[i]).all():
            order = np.argsort(xa[i])
            out[i] = np.interp(di[i], xa[i][order], k[i][order]) * di[i]
    return pd.DataFrame(out, index=ret.index, columns=ret.columns)


def hedged_return(ret: pd.DataFrame, duration: pd.DataFrame, anchors=HEDGE_ANCHORS) -> pd.DataFrame:
    """`ret` minus the duration-matched Treasury return (the anchors themselves hedge to zero)."""
    return ret - rates_return(ret, duration, anchors)
