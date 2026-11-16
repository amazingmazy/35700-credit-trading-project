"""Pre-trade risk: turn a strategy's raw targets into compliant targets using ONLY decision-time data.

Checks, in order:
  0. Blocked symbols (data-quality exclusions, optional) -> target 0, logged as blocked.
  1. Normalise: omitted symbols -> 0, NaN / non-finite -> keep current position.
  2. Untradable symbols (no valid 3PM price) -> keep current position (order rejected).
  3. Gross target: scale the book so its 3PM gross <= config.gmv_target.
  4. Worst-case gross (optional, default on): because the 3% volume cap may block any order,
     assume gross-increasing legs fill fully and gross-reducing legs fill nothing; shrink the
     gross-increasing legs so this worst case stays <= gmv_target.
The remaining risk, a 3PM -> close price move larger than the buffer, is monitored post-trade.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import BacktestConfig


@dataclass
class RiskResult:
    targets: pd.Series
    rejected: list[str] = field(default_factory=list)  # no valid 3PM price
    gross_scale: float = 1.0
    worst_case_scale: float = 1.0
    blocked: list[str] = field(default_factory=list)  # excluded for data-quality reasons


class RiskManager:
    def __init__(self, config: BacktestConfig):
        self.config = config

    @staticmethod
    def normalize_targets(raw: pd.Series | None, positions: pd.Series) -> pd.Series:
        symbols = positions.index
        if raw is None:
            return positions.copy()
        raw = raw[~raw.index.duplicated(keep="last")]
        unknown = raw.index.difference(symbols)
        if len(unknown):
            raise KeyError(f"Targets contain unknown symbols: {list(unknown)[:5]}")
        arr = raw.astype(float).reindex(symbols).to_numpy(copy=True)
        arr[~symbols.isin(raw.index)] = 0.0  # omitted -> flat
        bad = ~np.isfinite(arr)  # NaN / inf -> hold
        arr[bad] = positions.to_numpy(dtype=float)[bad]
        return pd.Series(arr, index=symbols)

    def check(
        self,
        raw_targets: pd.Series | None,
        positions: pd.Series,
        price: pd.Series,
        blocked: np.ndarray | None = None,
    ) -> RiskResult:
        """`blocked` is an optional boolean array over positions.index: symbols that must not be held
        (data-quality exclusions). Their targets are forced to zero before the other checks."""
        cfg = self.config
        t = self.normalize_targets(raw_targets, positions).to_numpy(copy=True)
        q = positions.to_numpy(dtype=float)
        p = price.reindex(positions.index).to_numpy(dtype=float)

        blocked_mask = np.zeros(len(t), dtype=bool)
        if blocked is not None:
            is_blocked = np.asarray(blocked, dtype=bool)
            blocked_mask = is_blocked & (t != 0)
            t[is_blocked] = 0.0

        tradable = np.isfinite(p) & (p > 0)
        rejected_mask = ~tradable & (t != q)
        t[~tradable] = q[~tradable]
        pz = np.where(tradable, p, 0.0)

        # 3. scale gross target
        gross = float(np.sum(np.abs(t) * pz))
        gross_scale = 1.0
        if gross > cfg.gmv_target:
            gross_scale = cfg.gmv_target / gross
            t = np.where(tradable, t * gross_scale, t)

        # 4. worst case: reductions fail, increases fill fully
        wc_scale = 1.0
        if cfg.worst_case_gmv_check:
            abs_q, abs_t = np.abs(q), np.abs(t)
            inc = np.where(tradable, np.maximum(abs_t - abs_q, 0.0), 0.0)
            base = float(np.sum(abs_q * pz))
            inc_dollars = float(np.sum(inc * pz))
            if base + inc_dollars > cfg.gmv_target and inc_dollars > 0:
                wc_scale = float(np.clip((cfg.gmv_target - base) / inc_dollars, 0.0, 1.0))
                grow = inc > 0
                new_abs = abs_q + wc_scale * inc
                t = np.where(grow, np.sign(t) * new_abs, t)
        targets = pd.Series(t, index=positions.index)
        return RiskResult(
            targets, list(positions.index[rejected_mask]), gross_scale, wc_scale, list(positions.index[blocked_mask])
        )

    def orders_from_targets(self, targets: pd.Series, positions: pd.Series) -> pd.Series:
        orders = targets - positions
        if self.config.whole_shares:
            orders = np.trunc(orders)
        return orders
