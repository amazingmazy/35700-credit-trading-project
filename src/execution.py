"""MOC execution at the official day-T close with the 3% volume cap and per-share costs."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import BacktestConfig
from src.order import Order, OrderState


@dataclass
class ExecutionReport:
    fills: pd.Series  # signed filled shares, full symbol index
    costs: pd.Series  # $ cost per symbol
    orders: list[Order]


class ExecutionHandler:
    def __init__(self, config: BacktestConfig, record_orders: bool = True):
        self.config = config
        self.record_orders = record_orders

    def max_fill(self, volume: pd.Series) -> pd.Series:
        v = volume.to_numpy(dtype=float)
        cap = np.where(np.isfinite(v) & (v > 0), self.config.participation_rate * v, 0.0)
        if self.config.whole_shares:
            cap = np.floor(cap + 1e-9)
        return pd.Series(cap, index=volume.index)

    def execute(
        self,
        orders: pd.Series,
        date: pd.Timestamp,
        close: pd.Series,
        volume: pd.Series,
        rejected: list[str] | None = None,
    ) -> ExecutionReport:
        idx = orders.index
        q = orders.to_numpy(dtype=float)
        px = close.reindex(idx).to_numpy(dtype=float)
        cap = self.max_fill(volume.reindex(idx)).to_numpy()
        priced = np.isfinite(px) & (px > 0)
        filled = np.sign(q) * np.minimum(np.abs(q), cap)
        filled = np.where(priced, filled, 0.0)
        costs = np.abs(filled) * self.config.cost_per_share

        order_log: list[Order] = []
        if self.record_orders:
            for sym in rejected or []:
                o = Order(sym, date, 0.0, reason="no valid 3PM price")
                o.transition(OrderState.REJECTED)
                order_log.append(o)
            for k in np.flatnonzero(q != 0):
                o = Order(idx[k], date, q[k])
                o.transition(OrderState.ACKED)
                o.filled_quantity = filled[k]
                o.cost = costs[k]
                o.fill_price = px[k]
                if filled[k] == 0:
                    o.reason = "no close price" if not priced[k] else "zero volume"
                    o.transition(OrderState.CANCELED)
                elif abs(filled[k]) < abs(q[k]):
                    o.reason = "volume cap; remainder cancelled"
                    o.transition(OrderState.PARTIALLY_FILLED)
                else:
                    o.transition(OrderState.FILLED)
                order_log.append(o)

        return ExecutionReport(pd.Series(filled, index=idx), pd.Series(costs, index=idx), order_log)
