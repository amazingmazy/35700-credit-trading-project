"""Positions, cash and daily mark-to-market accounting.

Daily sequence for date T:
  1. apply_corporate_actions: positions held at the T-1 close receive cash distributions
     going ex on T (shorts pay them), and splits rescale share counts and the reference price.
  2. apply_fills: trade at close_T, paying cost_per_share on every traded share.
  3. mark: value positions at close_T (last valid close if missing).
Equity_T = cash + sum(q * mark). PnL_T = Equity_T - Equity_{T-1}
         = price PnL + distributions - costs (exact identity, checked in tests).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import BacktestConfig


class Portfolio:
    def __init__(self, symbols: pd.Index, config: BacktestConfig):
        self.symbols = symbols
        self.config = config
        n = len(symbols)
        self.q = np.zeros(n)
        self.cash = float(config.initial_capital)
        self.mark_px = np.full(n, np.nan)  # last valid close
        self.equity = float(config.initial_capital)
        self._reset_day()

    def _reset_day(self) -> None:
        self.day_dividends = 0.0
        self.day_costs = 0.0
        self.day_traded_shares = 0.0
        self.day_traded_notional = 0.0

    # --------------------------------------------------------------------- views
    @property
    def positions(self) -> pd.Series:
        return pd.Series(self.q.copy(), index=self.symbols)

    def _value(self, px: np.ndarray) -> np.ndarray:
        return np.where(self.q != 0, self.q * np.nan_to_num(px), 0.0)

    def gmv(self) -> float:
        return float(np.sum(np.abs(self._value(self.mark_px))))

    def long_mv(self) -> float:
        v = self._value(self.mark_px)
        return float(v[v > 0].sum())

    def short_mv(self) -> float:
        v = self._value(self.mark_px)
        return float(v[v < 0].sum())

    def dv01(self, dv01_per_share_field: np.ndarray) -> float:
        """$ gain for a 1bp yield tightening (longs positive)."""
        d = np.nan_to_num(dv01_per_share_field) * self.config.dv01_scale
        return float(np.sum(np.where(self.q != 0, self.q * d, 0.0)))

    # ------------------------------------------------------------------ updates
    def apply_corporate_actions(self, split_ratio: np.ndarray, cash_per_share: np.ndarray) -> float:
        self._reset_day()
        div = float(np.sum(np.where(self.q != 0, self.q * cash_per_share, 0.0)))
        self.cash += div
        self.day_dividends = div
        if np.any(split_ratio != 1.0):
            self.q = self.q * split_ratio
            self.mark_px = self.mark_px / split_ratio
        # start-of-day book and reference prices, used for the independent price-PnL calculation
        self._q_open = self.q.copy()
        self._px_open = self.mark_px.copy()
        return div

    def apply_fills(self, fills: np.ndarray, close: np.ndarray, costs: np.ndarray) -> None:
        traded = fills != 0
        if not np.any(traded):
            return
        notional = np.where(traded, fills * close, 0.0)
        self.cash -= float(notional.sum()) + float(costs.sum())
        self.q = self.q + fills
        self.day_costs = float(costs.sum())
        self.day_traded_shares = float(np.abs(fills).sum())
        self.day_traded_notional = float(np.abs(notional).sum())
        self._fills = fills.copy()
        self._fill_px = np.where(traded, close, 0.0)

    def mark(self, close: np.ndarray) -> dict:
        """Mark to close; returns the day's accounting record (excluding date).

        pnl is the equity change (cash-flow based). price_pnl is computed independently from
        holdings x price changes, so pnl == price_pnl + dividends - costs is a real check.
        """
        prev_equity = self.equity
        new_px = np.where(np.isfinite(close) & (close > 0), close, self.mark_px)
        self.mark_px = new_px
        self.equity = self.cash + float(self._value(new_px).sum())
        pnl = self.equity - prev_equity

        q_open = getattr(self, "_q_open", np.zeros_like(self.q))
        px_open = getattr(self, "_px_open", new_px)
        held = q_open != 0
        price_pnl = float(np.sum(np.where(held, q_open * (np.nan_to_num(new_px) - np.nan_to_num(px_open)), 0.0)))
        fills = getattr(self, "_fills", None)
        if fills is not None:
            traded = fills != 0
            price_pnl += float(np.sum(np.where(traded, fills * (np.nan_to_num(new_px) - self._fill_px), 0.0)))
            self._fills = None
        return {
            "pnl": pnl,
            "dividends": self.day_dividends,
            "costs": self.day_costs,
            "price_pnl": price_pnl,
            "equity": self.equity,
            "cash": self.cash,
            "traded_shares": self.day_traded_shares,
            "traded_notional": self.day_traded_notional,
        }
