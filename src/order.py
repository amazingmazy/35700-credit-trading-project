"""Order lifecycle for daily MOC orders (style follows the Assignment-4 order state machine)."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import pandas as pd


class OrderState(Enum):
    NEW = auto()
    ACKED = auto()  # passed risk checks, sent as MOC
    REJECTED = auto()  # failed risk checks (e.g. no 3PM price)
    FILLED = auto()  # fully filled at the close
    PARTIALLY_FILLED = auto()  # volume cap bound; unfilled remainder cancelled (terminal)
    CANCELED = auto()  # nothing filled (no volume / no close)


TERMINAL_STATES = frozenset({OrderState.REJECTED, OrderState.FILLED, OrderState.PARTIALLY_FILLED, OrderState.CANCELED})

_ALLOWED_TRANSITIONS = {
    OrderState.NEW: {OrderState.ACKED, OrderState.REJECTED},
    OrderState.ACKED: {OrderState.FILLED, OrderState.PARTIALLY_FILLED, OrderState.CANCELED},
}


class InvalidTransition(Exception):
    pass


@dataclass
class Order:
    symbol: str
    date: pd.Timestamp
    quantity: float  # signed shares: + buy, - sell
    state: OrderState = OrderState.NEW
    filled_quantity: float = 0.0
    fill_price: float = float("nan")
    cost: float = 0.0
    reason: str = ""

    @property
    def side(self) -> str:
        return "buy" if self.quantity > 0 else "sell"

    def transition(self, new_state: OrderState) -> None:
        allowed = _ALLOWED_TRANSITIONS.get(self.state, set())
        if new_state not in allowed:
            raise InvalidTransition(f"{self.symbol}: {self.state.name} -> {new_state.name} not allowed")
        self.state = new_state

    def to_dict(self) -> dict:
        return {
            "date": self.date, "symbol": self.symbol, "quantity": self.quantity,
            "filled_quantity": self.filled_quantity, "fill_price": self.fill_price,
            "cost": self.cost, "state": self.state.name, "reason": self.reason,
        }
