"""Trivial strategies used to validate the engine."""
from __future__ import annotations

import pandas as pd

from src.context import DecisionContext
from src.strategies.base import Strategy, WeightStrategy, register


@register("no_trade")
class NoTrade(Strategy):
    """Never trades. PnL must be identically zero."""

    def generate_targets(self, ctx: DecisionContext) -> pd.Series:
        return pd.Series(0.0, index=ctx.symbols)


@register("buy_and_hold")
class BuyAndHold(Strategy):
    """Buy a fixed dollar notional of one ETF on the first tradable day, then hold that share count."""

    def __init__(self, ticker_or_symbol: str = "AGG", notional: float = 10_000_000.0):
        self.ticker_or_symbol = ticker_or_symbol
        self.notional = notional
        self._target: pd.Series | None = None

    def generate_targets(self, ctx: DecisionContext) -> pd.Series:
        if self._target is None:
            sym = self.ticker_or_symbol
            if sym not in ctx.symbols:
                sym = ctx.symbol_for(sym)
            p = ctx.price.get(sym)
            if p is None or not p > 0:
                return pd.Series(0.0, index=ctx.symbols)
            self._target = pd.Series(0.0, index=ctx.symbols)
            self._target[sym] = self.notional / p
        return self._target

    def signal(self, ctx: DecisionContext) -> pd.Series | None:
        return None


@register("equal_weight_long")
class EqualWeightLong(WeightStrategy):
    """Equal $ weight across the point-in-time universe, sized to config.gmv_target."""

    def __init__(self, min_history: int = 20, gross_fraction: float = 1.0):
        self.min_history = min_history
        self.gross_fraction = gross_fraction

    def target_weights(self, ctx: DecisionContext) -> pd.Series:
        uni = ctx.universe(self.min_history)
        if len(uni) == 0:
            return pd.Series(dtype=float)
        return pd.Series(self.gross_fraction / len(uni), index=uni)


@register("momentum_ls")
class MomentumLongShort(WeightStrategy):
    """Example signal strategy (NOT a proposed alpha): long top / short bottom quintile of trailing total return.

    Demonstrates history(), signal() for IC analysis and a dollar-neutral book.
    """

    def __init__(self, lookback: int = 60, quantile: float = 0.2, min_history: int = 60, gross_fraction: float = 1.0):
        self.lookback = lookback
        self.quantile = quantile
        self.min_history = min_history
        self.gross_fraction = gross_fraction
        self.warmup = lookback

    def signal(self, ctx: DecisionContext) -> pd.Series | None:
        tr = ctx.history("total_return", self.lookback)
        if len(tr) < self.lookback:
            return None
        score = (1 + tr).prod(min_count=int(self.lookback * 0.8)) - 1
        return score.reindex(ctx.universe(self.min_history)).dropna()

    def target_weights(self, ctx: DecisionContext) -> pd.Series:
        s = self.signal(ctx)
        if s is None or len(s) < 5:
            return pd.Series(dtype=float)
        lo, hi = s.quantile(self.quantile), s.quantile(1 - self.quantile)
        longs, shorts = s.index[s >= hi], s.index[s <= lo]
        w = pd.Series(0.0, index=s.index)
        w[longs] = 0.5 * self.gross_fraction / len(longs)
        w[shorts] = -0.5 * self.gross_fraction / len(shorts)
        return w
