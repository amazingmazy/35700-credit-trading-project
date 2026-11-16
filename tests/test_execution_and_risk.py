import numpy as np
import pandas as pd
import pytest

from src.config import BacktestConfig
from src.execution import ExecutionHandler
from src.order import InvalidTransition, Order, OrderState
from src.risk import RiskManager

IDX = pd.Index(["A", "B", "C"])
DATE = pd.Timestamp("2024-01-03")


def s(*vals):
    return pd.Series(vals, index=IDX, dtype=float)


class TestOrderStateMachine:
    def test_happy_paths(self):
        for final in (OrderState.FILLED, OrderState.PARTIALLY_FILLED, OrderState.CANCELED):
            o = Order("A", DATE, 10)
            o.transition(OrderState.ACKED)
            o.transition(final)
            assert o.state == final

    def test_rejected_from_new(self):
        o = Order("A", DATE, -5)
        o.transition(OrderState.REJECTED)
        assert o.state == OrderState.REJECTED and o.side == "sell"

    @pytest.mark.parametrize("bad", [OrderState.FILLED, OrderState.CANCELED, OrderState.PARTIALLY_FILLED])
    def test_cannot_fill_without_ack(self, bad):
        with pytest.raises(InvalidTransition):
            Order("A", DATE, 10).transition(bad)

    def test_terminal_states(self):
        o = Order("A", DATE, 10)
        o.transition(OrderState.ACKED)
        o.transition(OrderState.FILLED)
        with pytest.raises(InvalidTransition):
            o.transition(OrderState.ACKED)

    def test_to_dict(self):
        d = Order("A", DATE, 10).to_dict()
        assert d["state"] == "NEW" and d["symbol"] == "A"


class TestExecution:
    def setup_method(self):
        self.ex = ExecutionHandler(BacktestConfig())

    def test_volume_cap_is_3pct_floored(self):
        r = self.ex.execute(s(1e6, -1e6, 10), DATE, s(100, 50, 20), s(10_050, 10_050, 10_050))
        assert list(r.fills) == [301, -301, 10]  # floor(0.03 * 10050) = 301

    def test_zero_or_nan_volume_no_fill(self):
        r = self.ex.execute(s(100, 100, 100), DATE, s(100, 50, 20), s(0, np.nan, 1e6))
        assert list(r.fills) == [0, 0, 100]
        states = {o.symbol: o.state for o in r.orders}
        assert states["A"] == OrderState.CANCELED and states["B"] == OrderState.CANCELED
        assert states["C"] == OrderState.FILLED

    def test_nan_close_no_fill(self):
        r = self.ex.execute(s(100, 0, 0), DATE, s(np.nan, 50, 20), s(1e6, 1e6, 1e6))
        assert r.fills["A"] == 0 and r.orders[0].reason == "no close price"

    def test_costs_symmetric_per_share(self):
        r = self.ex.execute(s(1000, -1000, 0), DATE, s(100, 50, 20), s(1e6, 1e6, 1e6))
        assert r.costs["A"] == pytest.approx(3.5) and r.costs["B"] == pytest.approx(3.5)
        assert r.costs["C"] == 0

    def test_cost_charged_on_filled_not_ordered(self):
        r = self.ex.execute(s(1e6, 0, 0), DATE, s(100, 50, 20), s(1000, 1e6, 1e6))
        assert r.fills["A"] == 30 and r.costs["A"] == pytest.approx(30 * 0.0035)
        assert r.orders[0].state == OrderState.PARTIALLY_FILLED

    def test_rejected_orders_logged(self):
        r = self.ex.execute(s(0, 0, 0), DATE, s(1, 1, 1), s(1, 1, 1), rejected=["B"])
        assert r.orders[0].state == OrderState.REJECTED

    def test_fractional_mode(self):
        ex = ExecutionHandler(BacktestConfig(whole_shares=False), record_orders=False)
        r = ex.execute(s(1e6, 0, 0), DATE, s(100, 50, 20), s(1010, 1, 1))
        assert r.fills["A"] == pytest.approx(30.3) and r.orders == []


class TestRiskManager:
    def setup_method(self):
        self.cfg = BacktestConfig(gmv_target=1_000_000, gmv_cap=1_200_000)
        self.rm = RiskManager(self.cfg)

    def test_normalize_omitted_is_zero_nan_is_hold(self):
        pos = s(10, 20, 30)
        t = self.rm.normalize_targets(pd.Series({"A": 5.0, "B": np.nan}), pos)
        assert list(t) == [5, 20, 0]
        assert self.rm.normalize_targets(None, pos).equals(pos)

    def test_unknown_symbol_raises(self):
        with pytest.raises(KeyError):
            self.rm.normalize_targets(pd.Series({"ZZZ": 1.0}), s(0, 0, 0))

    def test_untradable_symbol_kept_and_rejected(self):
        r = self.rm.check(pd.Series({"A": 100.0, "B": 0.0}), s(0, 50, 0), s(10, np.nan, 10))
        assert r.targets["B"] == 50 and r.rejected == ["B"]

    def test_gross_scaled_to_target(self):
        r = self.rm.check(s(10_000, -10_000, 0), s(0, 0, 0), s(100, 100, 100))  # 2MM gross
        gross = (r.targets.abs() * 100).sum()
        assert gross == pytest.approx(1_000_000) and r.gross_scale == pytest.approx(0.5)

    def test_worst_case_rotation_limited(self):
        # fully invested in A, want to rotate to B: if the A sale fails, B buy must not breach target
        pos = s(10_000, 0, 0)  # $1MM at $100
        r = self.rm.check(s(0, 10_000, 0), pos, s(100, 100, 100))
        worst = (np.maximum(pos.abs(), r.targets.abs()) * 100).sum()
        assert worst <= 1_000_000 + 1e-6
        assert r.targets["A"] == 0  # reductions untouched
        assert r.worst_case_scale == pytest.approx(0.0)

    def test_worst_case_disabled(self):
        rm = RiskManager(BacktestConfig(gmv_target=1_000_000, gmv_cap=1_200_000, worst_case_gmv_check=False))
        r = rm.check(s(0, 10_000, 0), s(10_000, 0, 0), s(100, 100, 100))
        assert r.targets["B"] == 10_000

    def test_orders_truncate_toward_zero(self):
        o = self.rm.orders_from_targets(s(10.7, -10.7, 3.0), s(0, 0, 0.5))
        assert list(o) == [10, -10, 2]


class TestConfig:
    @pytest.mark.parametrize("kw", [dict(initial_capital=0), dict(gmv_target=70e6), dict(participation_rate=0),
                                    dict(cost_per_share=-1)])
    def test_invalid(self, kw):
        with pytest.raises(ValueError):
            BacktestConfig(**kw)

    def test_defaults_match_project_rules(self):
        c = BacktestConfig()
        assert (c.initial_capital, c.gmv_cap, c.participation_rate, c.cost_per_share, c.periods_per_year) == \
            (10_000_000, 60_000_000, 0.03, 0.0035, 252)
