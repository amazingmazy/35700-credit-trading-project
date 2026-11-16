"""Backtest configuration: capital, risk limits, execution and reporting conventions."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BacktestConfig:
    """All project-wide rules in one immutable object.

    Defaults follow the FINM 35700 project statement.
    """

    # Capital (fixed; returns are PnL / initial_capital, no compounding)
    initial_capital: float = 10_000_000.0

    # Gross exposure: hard cap from the rules, and the lower level we size to at 3PM.
    # The buffer absorbs 3PM -> close price moves we cannot see at decision time.
    gmv_cap: float = 60_000_000.0
    gmv_target: float = 57_000_000.0

    # If True, size orders so the GMV stays under gmv_target even in the worst case: every
    # gross-increasing order fills and every gross-reducing order fails to fill (volume cap).
    worst_case_gmv_check: bool = True

    # Execution (MOC at official close)
    participation_rate: float = 0.03  # max fill = 3% of day-T share volume
    cost_per_share: float = 0.0035  # charged on every share actually traded
    whole_shares: bool = True  # targets/orders are truncated toward zero

    # Reporting
    periods_per_year: int = 252
    risk_free_rate: float = 0.0
    # Per-share $DV01 = dv01_field * dv01_scale. Empirically YAS_RISK ~= close * mod_duration / 100,
    # i.e. the price change for a 100bp move, so $ per 1bp = YAS_RISK / 100.
    dv01_scale: float = 0.01

    def __post_init__(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if not 0 < self.gmv_target <= self.gmv_cap:
            raise ValueError("require 0 < gmv_target <= gmv_cap")
        if not 0 < self.participation_rate <= 1:
            raise ValueError("participation_rate must be in (0, 1]")
        if self.cost_per_share < 0:
            raise ValueError("cost_per_share must be non-negative")
