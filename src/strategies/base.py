"""Strategy interface. Adding a new strategy = subclass Strategy (or WeightStrategy) + @register."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import pandas as pd

from src.context import DecisionContext

STRATEGY_REGISTRY: dict[str, type["Strategy"]] = {}


def register(name: str) -> Callable[[type["Strategy"]], type["Strategy"]]:
    """Class decorator: make a strategy available to main.py --strategy <name>."""

    def deco(cls: type["Strategy"]) -> type["Strategy"]:
        if name in STRATEGY_REGISTRY:
            raise ValueError(f"Strategy {name!r} already registered")
        STRATEGY_REGISTRY[name] = cls
        cls.name = name
        return cls

    return deco


class Strategy(ABC):
    """Base strategy.

    generate_targets(ctx) returns the FULL desired book in shares, indexed by symbol:
      * symbols omitted from the Series -> target 0 (flatten)
      * NaN target                       -> keep the current position (no order)
    Optionally override signal(ctx) to return a cross-sectional score for IC / Rank IC analysis.
    """

    name: str = "base"
    warmup: int = 0  # days with no trading at the start (reported separately)

    @abstractmethod
    def generate_targets(self, ctx: DecisionContext) -> pd.Series:
        ...

    def signal(self, ctx: DecisionContext) -> pd.Series | None:
        return None

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class WeightStrategy(Strategy):
    """Convenience base: return weights as a fraction of config.gmv_target gross $.

    sum(|w|) <= 1 keeps the book within the gross-exposure target.
    """

    @abstractmethod
    def target_weights(self, ctx: DecisionContext) -> pd.Series:
        ...

    def generate_targets(self, ctx: DecisionContext) -> pd.Series:
        return ctx.weights_to_shares(self.target_weights(ctx))
