"""End-to-end accounting tests on synthetic data."""
import numpy as np
import pandas as pd
import pytest

from src.backtester import Backtester
from src.config import BacktestConfig
from src.strategies import STRATEGY_REGISTRY, BuyAndHold, EqualWeightLong, MomentumLongShort, NoTrade
from src.strategies.base import Strategy, register
from src.data_loader import MarketData
from tests.conftest import (ALL_DATES, TRADING_DATES, FixedTargets, base_prices, corp_event, make_frames, make_market,
                            vectorized_price_plus_dividend_pnl)


def run(market, strategy, **cfg):
    return Backtester(market, strategy, BacktestConfig(**cfg)).run()


class LongShortRotate(Strategy):
    """Long/short book that changes every day, so it trades through dividends and splits."""

    def generate_targets(self, ctx):
        k = ctx.n_history
        w = pd.Series({"A": 0.4, "B": -0.3 if k % 2 else 0.3, "C": -0.3 if k % 3 else 0.2})
        return ctx.weights_to_shares(w, gross=2e6)


class TestIndependentLedgerCheck:
    """Ledger PnL (cash-flow based) vs. a vectorized holdings x total-return calculation."""

    def test_matches_vectorized_with_dividends_and_split(self):
        ex_split, ex_div = TRADING_DATES[5], TRADING_DATES[3]
        p = base_prices()
        p["B"][list(ALL_DATES).index(ex_split):] = 25.0
        m = make_market(prices=p, volume=1e8, corp_rows=[
            corp_event("B", ex_split, 2.0, "Stock Split"),
            corp_event("A", ex_div, 0.75), corp_event("C", ex_div, 0.10)])
        r = run(m, LongShortRotate())
        d = r.daily
        alt = vectorized_price_plus_dividend_pnl(m, r.positions)
        np.testing.assert_allclose(d["price_pnl"] + d["dividends"], alt, atol=1e-6)
        np.testing.assert_allclose(d["pnl"], d["price_pnl"] + d["dividends"] - d["costs"], atol=1e-6)
        assert d.loc[ex_div, "dividends"] != 0 and d["costs"].sum() > 0


class TestTrivialStrategy:
    def test_no_trade_zero_pnl_every_day(self, market):
        r = run(market, NoTrade())
        assert len(r.daily) == len(TRADING_DATES)  # inactive days still recorded
        assert (r.daily["pnl"] == 0).all() and (r.daily["gmv"] == 0).all()
        m = r.metrics()
        assert np.isnan(m["sharpe"]) and m["max_drawdown_usd"] == 0 and m["total_costs"] == 0
        assert r.orders.empty

    def test_no_trade_ignores_dividends(self):
        m = make_market(corp_rows=[corp_event("A", TRADING_DATES[3], 1.0)])
        assert (run(m, NoTrade()).daily["pnl"] == 0).all()


class TestAccounting:
    def test_buy_and_hold_pnl_by_hand(self, market):
        # buy 1000 A on day 0 at close 100; A rises $1 per business day (holiday included in the gap)
        r = run(market, FixedTargets({"A": 1000}))
        d = r.daily
        assert d["pnl"].iloc[0] == pytest.approx(-1000 * 0.0035)  # trade at close, cost only
        closes = market.eod["close"]["A"]
        expected = 1000 * closes.diff().iloc[1:]
        np.testing.assert_allclose(d["pnl"].iloc[1:], expected, rtol=1e-12)
        assert r.positions["A"].iloc[-1] == 1000

    def test_pnl_identity(self, market):
        r = run(market, EqualWeightLong(min_history=2))
        d = r.daily
        assert d["pnl"].sum() == pytest.approx(d["equity"].iloc[-1] - 10_000_000, abs=1e-6)
        np.testing.assert_allclose(d["pnl"], d["price_pnl"] + d["dividends"] - d["costs"], atol=1e-6)
        assert d["costs"].sum() == pytest.approx(0.0035 * d["traded_shares"].sum())

    def test_returns_on_fixed_capital(self, market):
        r = run(market, EqualWeightLong(min_history=2))
        np.testing.assert_allclose(r.daily["return"], r.daily["pnl"] / 10_000_000)

    def test_partial_fills_accumulate_over_days(self):
        m = make_market(volume={"A": 10_000.0})  # 300 shares/day
        r = run(m, FixedTargets({"A": 1000}))
        assert list(r.positions["A"].iloc[:4]) == [300, 600, 900, 1000]
        assert (r.orders["state"].iloc[:3] == "PARTIALLY_FILLED").all()

    def test_warmup_days_flagged_and_inactive(self, market):
        r = run(market, MomentumLongShort(lookback=3, min_history=3))
        assert r.daily["warmup"].sum() == 3
        assert (r.daily.loc[r.daily["warmup"], "traded_shares"] == 0).all()


class TestCorporateActionsInPnL:
    def test_dividend_long_receives_short_pays_on_ex_date_only(self):
        ex = TRADING_DATES[4]
        m = make_market(corp_rows=[corp_event("B", ex, 0.25)])  # B flat at 50 -> PnL is only dividends
        r = run(m, FixedTargets({"B": 1000, "A": 0}), worst_case_gmv_check=False)
        r_short = run(m, FixedTargets({"B": -1000}))
        assert r.daily.loc[ex, "dividends"] == pytest.approx(250.0)
        assert r_short.daily.loc[ex, "dividends"] == pytest.approx(-250.0)
        assert r.daily["dividends"].drop(ex).abs().sum() == 0
        assert r.daily.loc[ex, "pnl"] == pytest.approx(250.0)

    def test_dividend_not_paid_if_bought_on_ex_date(self):
        ex = TRADING_DATES[0]
        m = make_market(corp_rows=[corp_event("B", ex, 0.25)])
        assert run(m, FixedTargets({"B": 1000})).daily["dividends"].sum() == 0

    def test_split_is_pnl_neutral(self):
        ex = TRADING_DATES[5]
        p = base_prices()
        p["B"][list(ALL_DATES).index(ex):] = 25.0  # 2:1 split
        m = make_market(prices=p, corp_rows=[corp_event("B", ex, 2.0, "Stock Split")])
        r = run(m, FixedTargets({"B": 1000}, hold_after_first=True))
        assert r.daily.loc[ex, "pnl"] == pytest.approx(0.0)
        assert r.positions.loc[ex, "B"] == 2000
        assert r.daily["traded_shares"].iloc[1:].sum() == 0
        assert r.daily["pnl"].iloc[1:].abs().sum() == pytest.approx(0.0)

    def test_split_without_adjustment_would_show_loss(self):
        ex = TRADING_DATES[5]
        p = base_prices()
        p["B"][list(ALL_DATES).index(ex):] = 25.0
        m = make_market(prices=p)  # no corp action row -> raw price drop shows up as a loss
        r = run(m, FixedTargets({"B": 1000}, hold_after_first=True))
        assert r.daily.loc[ex, "pnl"] == pytest.approx(-25_000)


class TestCapitalLimits:
    def test_gmv_never_exceeds_cap(self):
        p = base_prices()
        p["A"] = p["A"] * np.linspace(1, 1.03, len(p["A"]))
        r = run(make_market(prices=p, volume=1e8), EqualWeightLong(min_history=1))
        assert (r.daily["gmv"] <= 60_000_000).all()
        assert r.daily["gmv"].max() > 50_000_000  # actually uses the capacity
        assert not r.daily["gmv_breach"].any()

    def test_oversized_targets_scaled(self, market):
        r = run(market, FixedTargets({"A": 10_000_000}))
        assert r.daily["gmv"].iloc[0] <= 57_000_000 * 1.001

    def test_risk_downsizes_held_book_when_3pm_gross_exceeds_target(self):
        p = base_prices()
        k = list(ALL_DATES).index(TRADING_DATES[1])
        p["A"][k:] = p["A"][k:] * 1.5  # +50% jump visible at 3PM on day 1
        r = run(make_market(prices=p, volume=1e8), FixedTargets({"A": 1e9}, hold_after_first=True))
        assert not r.daily["gmv_breach"].any()
        assert r.positions["A"].iloc[1] < r.positions["A"].iloc[0]

    def test_breach_is_flagged_when_close_jumps_after_3pm(self):
        eod, snap, corp, sym = make_frames(volume=1e8)
        d1 = TRADING_DATES[1]
        eod.loc[(eod.symbol == "A") & (eod.date == d1), "close"] *= 1.5  # move invisible at 3PM
        m = MarketData.from_frames(eod, snap, corp, sym)
        r = run(m, FixedTargets({"A": 1e9}, hold_after_first=True))
        assert r.daily["gmv_breach"].iloc[1] and not r.daily["gmv_breach"].iloc[0]
        assert r.metrics()["gmv_breach_days"] >= 1


class TestDV01:
    def test_dv01_sign_and_units(self, market):
        # 1000 shares A at 100 with duration 5 -> $DV01 = 1000 * 100 * 5 / 10000 = $50 per bp
        r = run(market, FixedTargets({"A": 1000}))
        assert r.daily["dv01"].iloc[0] == pytest.approx(50.0)
        r_short = run(market, FixedTargets({"A": -1000}))
        assert r_short.daily["dv01"].iloc[0] == pytest.approx(-50.0)


class TestStrategyPlumbing:
    def test_registry(self):
        assert {"no_trade", "buy_and_hold", "equal_weight_long", "momentum_ls"} <= set(STRATEGY_REGISTRY)
        with pytest.raises(ValueError):
            register("no_trade")(NoTrade)

    def test_new_strategy_injection(self, market):
        class LongC(Strategy):
            def generate_targets(self, ctx):
                return pd.Series({"C": 100.0})

        r = run(market, LongC())
        assert r.positions["C"].iloc[-1] == 100 and (r.positions[["A", "B"]] == 0).all().all()
        assert "LongC" in repr(LongC())

    def test_buy_and_hold_by_ticker(self, market):
        r = run(market, BuyAndHold("BBB", notional=50_000))
        assert r.positions["B"].iloc[-1] == pytest.approx(1001)  # trunc(50_000 / 49.95)

    def test_buy_and_hold_waits_for_price(self):
        m = make_market(snap_last_nan=[("A", TRADING_DATES[0])])
        m.snap["price"].loc[TRADING_DATES[0], "A"] = np.nan
        r = run(m, BuyAndHold("A", notional=10_000))
        assert r.positions["A"].iloc[0] == 0 and r.positions["A"].iloc[1] > 0

    def test_momentum_signals_recorded(self, market):
        r = run(market, MomentumLongShort(lookback=3, min_history=3))
        assert not r.signals.empty
        assert r.signals.index.min() >= TRADING_DATES[3]
