"""Information-set tests: a strategy at 3:45PM on T can only see EOD <= T-1, 3PM(T), declared corp actions."""
import copy

import numpy as np
import pandas as pd
import pytest

from src.backtester import Backtester
from src.config import BacktestConfig
from src.context import DecisionContext
from src.strategies.base import Strategy
from tests.conftest import TRADING_DATES, corp_event, make_market


def ctx_at(market, i, config=None):
    return DecisionContext(market, i, pd.Series(0.0, index=market.symbols), 10e6, config or BacktestConfig())


class TestDecisionContext:
    def test_history_strictly_before_T(self, market):
        for i in (0, 1, 5, len(market.dates) - 1):
            h = ctx_at(market, i).history("close")
            assert len(h) == i
            if i:
                assert h.index.max() < market.dates[i]

    def test_history_lookback(self, market):
        h = ctx_at(market, 8).history("close", lookback=3)
        assert list(h.index) == list(market.dates[5:8])

    def test_history_is_a_copy(self, market):
        ctx = ctx_at(market, 5)
        h = ctx.history("close")
        h.iloc[:, :] = -1
        assert (market.eod["close"].iloc[:5] > 0).all().all()

    def test_eod_row_lag_zero_forbidden(self, market):
        ctx = ctx_at(market, 3)
        with pytest.raises(ValueError):
            ctx.eod_row("close", lag=0)
        assert ctx.eod_row("close", 1).equals(market.eod["close"].iloc[2])
        assert ctx_at(market, 0).eod_row("close").isna().all()

    def test_total_return_history(self, market):
        h = ctx_at(market, 4).history("total_return")
        assert h.index.max() < market.dates[4]
        assert ctx_at(market, 4).eod_row("total_return").equals(market.total_return.iloc[3])

    def test_snapshot_is_day_T(self, market):
        ctx = ctx_at(market, 4)
        snap = ctx.snapshot()
        assert set(snap.columns) >= {"last", "bid", "ask", "mid", "price"}
        assert ctx.price.equals(market.snap["price"].iloc[4])

    def test_known_corp_actions_use_declared_date(self):
        m = make_market(corp_rows=[corp_event("A", TRADING_DATES[6], 0.5, declared=TRADING_DATES[4])])
        assert ctx_at(m, 3).known_corp_actions().empty
        assert len(ctx_at(m, 4).known_corp_actions()) == 1

    def test_universe_point_in_time(self, market):
        assert len(ctx_at(market, 2).universe(min_history=3)) == 0
        assert list(ctx_at(market, 3).universe(min_history=3)) == ["A", "B", "C"]
        assert len(ctx_at(market, 0).universe(min_history=0)) == 3

    def test_weights_to_shares(self, market, config):
        ctx = ctx_at(market, 2, config)
        shares = ctx.weights_to_shares(pd.Series({"A": 0.5, "B": -0.5}))
        assert shares["A"] == pytest.approx(0.5 * config.gmv_target / ctx.price["A"])
        assert shares["B"] < 0 and shares["C"] == 0


class HistoryHungry(Strategy):
    """Uses every allowed input so poisoning anything forbidden would change its output."""

    name = "history_hungry"

    def generate_targets(self, ctx):
        h = ctx.history("close")
        v = ctx.history("volume")
        if len(h) < 2:
            return pd.Series(0.0, index=ctx.symbols)
        score = h.iloc[-1] / h.iloc[0] + v.mean() * 1e-9 + ctx.price * 1e-6
        w = (score - score.mean())
        w = w / w.abs().sum()
        return ctx.weights_to_shares(w, gross=1e6)


class TestNoLookahead:
    @pytest.mark.parametrize("k", [1, 4, 8])
    def test_poisoning_day_T_and_future_does_not_change_decisions(self, market, k):
        base = Backtester(market, HistoryHungry(), record_targets=True).run()

        poisoned = copy.deepcopy(market)
        for name, panel in poisoned.eod.items():
            panel.iloc[k:] = 1e9 if name != "volume" else 0.0  # day-T close/volume and all later EOD
        poisoned.total_return.iloc[k:] = 5.0
        for panel in poisoned.snap.values():
            panel.iloc[k + 1:] = 1e9  # future 3PM snapshots (day-T snapshot is allowed)
        bad = Backtester(poisoned, HistoryHungry(), record_targets=True).run()

        pd.testing.assert_frame_equal(base.targets.iloc[: k + 1], bad.targets.iloc[: k + 1])
        assert not base.targets.iloc[k + 1:].equals(bad.targets.iloc[k + 1:])
