"""DecisionContext: the ONLY view of the world a strategy receives at 3:45PM on day T.

Information rule (project statement section 3), enforced structurally:
  1. EOD observations through T-1      -> history(), eod_row(lag>=1)
  2. Day-T 3PM snapshot                -> snapshot(), price
  3. Corp actions known at submission  -> known_corp_actions() (declared_date <= T)
Day-T close and volume are never exposed here.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.config import BacktestConfig

if TYPE_CHECKING:  # pragma: no cover
    from src.data_loader import MarketData


class DecisionContext:
    def __init__(
        self,
        market: "MarketData",
        t_index: int,
        positions: pd.Series,
        equity: float,
        config: BacktestConfig,
    ):
        self._market = market
        self._i = t_index
        self.date: pd.Timestamp = market.dates[t_index]
        self.symbols: pd.Index = market.symbols
        self.positions: pd.Series = positions.copy()  # shares held entering T (after day-T splits)
        self.equity: float = equity  # $ equity at T-1 close + day-T distributions
        self.config = config

    # ---------------------------------------------------------------- EOD (<= T-1)
    @property
    def n_history(self) -> int:
        """Number of EOD rows available (dates strictly before T)."""
        return self._i

    def history(self, field: str, lookback: int | None = None) -> pd.DataFrame:
        """EOD panel for `field` over dates < T (copy). `total_return` is also available."""
        panel = self._market.total_return if field == "total_return" else self._market.eod[field]
        start = 0 if lookback is None else max(0, self._i - lookback)
        return panel.iloc[start:self._i].copy()

    def eod_row(self, field: str, lag: int = 1) -> pd.Series:
        """EOD values `lag` trading days before T (lag >= 1). NaN-filled if unavailable."""
        if lag < 1:
            raise ValueError("lag must be >= 1: day-T EOD data is not available at 3:45PM")
        panel = self._market.total_return if field == "total_return" else self._market.eod[field]
        j = self._i - lag
        if j < 0:
            return pd.Series(np.nan, index=self.symbols)
        return panel.iloc[j].copy()

    # ---------------------------------------------------------------- 3PM (T)
    def snapshot(self) -> pd.DataFrame:
        """Day-T 3PM snapshot: columns last, bid, ask, mid, price (decision price)."""
        return pd.DataFrame({k: v.iloc[self._i] for k, v in self._market.snap.items()})

    @property
    def price(self) -> pd.Series:
        """Day-T 3PM decision price (last, else mid)."""
        return self._market.snap["price"].iloc[self._i].copy()

    # ---------------------------------------------------------------- corp actions
    def known_corp_actions(self) -> pd.DataFrame:
        ca = self._market.corp_actions
        return ca[ca["declared_date"] <= self.date].copy()

    # ---------------------------------------------------------------- helpers
    def symbol_for(self, ticker: str) -> str:
        """Ticker -> ISIN (static reference data)."""
        return self._market.symbol_for(ticker)

    def tradable(self) -> pd.Series:
        """Boolean mask: valid 3PM price today."""
        p = self.price
        return p.notna() & (p > 0)

    def universe(self, min_history: int = 20) -> pd.Index:
        """Point-in-time universe: >= min_history EOD closes through T-1 and a valid 3PM price.

        With config.clean_universe_only (default) symbols with missing dividend records or an unrecorded
        split are left out. That exclusion list comes from the full sample (see src/data_quality.py).
        """
        counts = self._market.history_count(self._i)
        ok = (counts >= min_history) & self.tradable().to_numpy()
        if self.config.clean_universe_only:
            ok &= ~self._market.excluded_mask()
        return self.symbols[ok]

    def weights_to_shares(self, weights: pd.Series, gross: float | None = None) -> pd.Series:
        """Convert weights (fraction of `gross` $, default config.gmv_target) to share targets at 3PM prices."""
        gross = self.config.gmv_target if gross is None else gross
        w = weights.reindex(self.symbols).fillna(0.0)
        p = self.price
        shares = (w * gross / p).where(p > 0, 0.0).fillna(0.0)
        return shares
