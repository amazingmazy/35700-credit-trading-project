import math

import numpy as np
import pandas as pd
import pytest

from src.backtester import Backtester
from src.config import BacktestConfig
from src.data_loader import MarketData
from src.performance import (SHARPE_AUDIT_THRESHOLD, PerformanceAnalyzer, bill_rate_from_market, drawdowns,
                             equity_curve, financing_charge, information_ratio, max_drawdown, sharpe_ratio)
from src.signal_analysis import (SignalAnalyzer, forward_returns, ic_summary, information_coefficient,
                                 newey_west_tstat)
from src.strategies import EqualWeightLong
from tests.conftest import TRADING_DATES, make_frames, make_market

C0 = 10_000_000


class TestRequiredMetrics:
    def test_sharpe_by_hand(self):
        r = pd.Series([0.01, -0.005, 0.002, 0.004])
        expected = math.sqrt(252) * r.mean() / r.std(ddof=1)
        assert sharpe_ratio(r) == pytest.approx(expected)
        assert r.std(ddof=1) != r.std(ddof=0)  # guards ddof choice

    def test_sharpe_undefined(self):
        assert np.isnan(sharpe_ratio(pd.Series([0.01])))
        assert np.isnan(sharpe_ratio(pd.Series([0.0, 0.0, 0.0])))
        assert np.isnan(sharpe_ratio(pd.Series([], dtype=float)))

    def test_drawdown_by_hand(self):
        pnl = pd.Series([-100.0, 50.0, -200.0, 300.0])
        # V = 9,999,900 / 9,999,950 / 9,999,750 / 10,000,050 ; M includes V0 = 10MM
        dd_usd, dd_pct = max_drawdown(pnl, C0)
        assert dd_usd == pytest.approx(250.0)
        assert dd_pct == pytest.approx(250.0 / C0)

    def test_drawdown_includes_v0(self):
        assert max_drawdown(pd.Series([-100.0]), C0)[0] == pytest.approx(100.0)

    def test_drawdown_positive_magnitudes(self):
        dd_usd, dd_pct = drawdowns(pd.Series([10.0, 20.0, -5.0]), C0)
        assert (dd_usd >= 0).all() and (dd_pct >= 0).all()
        assert max_drawdown(pd.Series([], dtype=float), C0) == (0.0, 0.0)

    def test_equity_curve(self):
        assert list(equity_curve(pd.Series([1.0, 2.0]), C0)) == [C0 + 1, C0 + 3]

    def test_required_metric_set(self):
        pnl = pd.Series([1000.0, -500.0, 2000.0])
        m = PerformanceAnalyzer().required_metrics(pnl)
        assert m["annualized_net_pnl"] == pytest.approx(252 * pnl.mean())
        assert m["sharpe"] == pytest.approx(sharpe_ratio(pnl / C0))
        assert m["n_days"] == 3

    def test_risk_free_rate_is_annualized(self):
        r = pd.Series([0.01, -0.005, 0.002, 0.004])
        assert sharpe_ratio(r, 252, rf=0.0252) == pytest.approx(sharpe_ratio(r - 0.0001))
        assert sharpe_ratio(r, 252, rf=0.0252) < sharpe_ratio(r)

    def test_information_ratio(self):
        r = pd.Series([0.01, 0.02, -0.01, 0.0])
        b = pd.Series([0.005, 0.01, -0.02, 0.001])
        assert information_ratio(r, b) == pytest.approx(sharpe_ratio(r - b))

    def test_summary_and_format(self, market):
        res = Backtester(market, EqualWeightLong(min_history=1)).run()
        bench = market.total_return["A"].fillna(0)
        m = res.metrics(bench)
        for k in ("sharpe", "max_drawdown_usd", "max_drawdown_pct", "annualized_net_pnl", "information_ratio",
                  "total_costs", "avg_daily_turnover", "avg_gmv", "avg_dv01", "gross_sharpe"):
            assert k in m
        assert m["gross_sharpe"] >= m["sharpe"] or np.isnan(m["sharpe"])
        text = PerformanceAnalyzer.format_summary(m)
        assert "sharpe" in text and "%" in text


class TestNetOfFinancing:
    """Working metric: PnL after charging the T-bill rate on the net $ exposure carried overnight."""

    def daily(self, long_mv, short_mv, pnl):
        idx = pd.bdate_range("2024-01-01", periods=len(pnl))
        return pd.DataFrame({"long_mv": long_mv, "short_mv": short_mv, "pnl": pnl}, index=idx)

    def test_financing_charge_by_hand(self):
        idx = pd.bdate_range("2024-01-01", periods=3)
        net = pd.Series([0.0, 25_200_000.0, 25_200_000.0], index=idx)
        rate = pd.Series(0.05, index=idx)
        charge = financing_charge(net, rate)
        # exposure is charged the day AFTER it is put on: 25.2MM x 5% / 252 = 5,000
        assert list(charge) == pytest.approx([0.0, 0.0, 5000.0])

    def test_long_only_book_loses_the_bill_rate(self):
        n = 50
        rng = np.random.default_rng(1)
        pnl = 60e6 * 0.05 / 252 + rng.normal(0, 500.0, n)  # earns exactly the bill rate plus noise
        d = self.daily(np.full(n, 60e6), np.zeros(n), pnl)
        m = PerformanceAnalyzer().financing_adjusted(d, pd.Series(0.05, index=d.index))
        assert m["annualized_financing_charge"] == pytest.approx(60e6 * 0.05 * (n - 1) / n)
        assert abs(m["sharpe_net_of_financing"]) < 3 < sharpe_ratio(d["pnl"] / C0)
        assert m["avg_net_exposure"] == pytest.approx(60e6)

    def test_dollar_neutral_book_is_unchanged(self):
        n = 30
        pnl = np.random.default_rng(2).normal(100.0, 1000.0, n)
        d = self.daily(np.full(n, 30e6), np.full(n, -30e6), pnl)
        m = PerformanceAnalyzer().financing_adjusted(d, pd.Series(0.05, index=d.index))
        assert m["annualized_financing_charge"] == 0.0
        assert m["sharpe_net_of_financing"] == pytest.approx(sharpe_ratio(d["pnl"] / C0))

    def test_audit_flag_above_threshold(self):
        n = 40
        pnl = 2000.0 + np.random.default_rng(3).normal(0, 100.0, n)
        d = self.daily(np.zeros(n), np.zeros(n), pnl)
        assert PerformanceAnalyzer().financing_adjusted(d, pd.Series(0.05, index=d.index))["sharpe_audit_required"]
        d2 = self.daily(np.zeros(n), np.zeros(n), np.random.default_rng(4).normal(0, 100.0, n))
        assert not PerformanceAnalyzer().financing_adjusted(d2, pd.Series(0.05, index=d2.index))["sharpe_audit_required"]
        assert SHARPE_AUDIT_THRESHOLD == 1.0

    def test_bill_rate_from_market_and_summary(self):
        eod, snap, corp, sym = make_frames()
        eod["yield"] = np.where(eod["symbol"] == "A", 5.0, 9.0)
        eod.loc[(eod["symbol"] == "A") & (eod["date"] == TRADING_DATES[4]), "yield"] = 12602.0  # outlier
        m = MarketData.from_frames(eod, snap, corp, sym)
        rate = bill_rate_from_market(m, ticker="AAA")
        assert rate.notna().all() and np.allclose(rate, 0.05)  # lagged, outlier dropped, leading gap filled
        assert bill_rate_from_market(m, ticker="ZZZ") is None
        assert bill_rate_from_market(make_market(), ticker="AAA") is None  # no yield field
        res = Backtester(m, EqualWeightLong(min_history=1)).run()
        out = res.metrics(bill_rate=rate)
        assert out["sharpe_net_of_financing"] < out["sharpe"] and out["annualized_financing_charge"] > 0
        text = PerformanceAnalyzer.format_summary(out)
        assert "sharpe_project_formula" in text and "sharpe_net_of_financing" in text
        assert "sharpe_net_of_financing" not in res.metrics()  # only reported when a bill rate is supplied


class TestSignalAnalysis:
    def setup_method(self):
        rng = np.random.default_rng(0)
        idx = pd.bdate_range("2024-01-01", periods=60)
        self.tr = pd.DataFrame(rng.normal(0, 0.01, (60, 8)), index=idx, columns=list("abcdefgh"))

    def test_forward_returns_compound(self):
        f2 = forward_returns(self.tr, 2)
        expected = (1 + self.tr.iloc[1]) * (1 + self.tr.iloc[2]) - 1
        np.testing.assert_allclose(f2.iloc[0], expected)
        assert f2.iloc[-2:].isna().all().all()
        with pytest.raises(ValueError):
            forward_returns(self.tr, 0)

    def test_perfect_foresight_ic_is_one(self):
        sig = forward_returns(self.tr, 1)
        ic = information_coefficient(sig, forward_returns(self.tr, 1))
        ric = information_coefficient(sig, forward_returns(self.tr, 1), rank=True)
        assert np.allclose(ic, 1.0) and np.allclose(ric, 1.0)

    def test_monotone_transform_keeps_rank_ic(self):
        fwd = forward_returns(self.tr, 1)
        sig = np.exp(200 * fwd)
        assert np.allclose(information_coefficient(sig, fwd, rank=True), 1.0)
        assert information_coefficient(sig, fwd).mean() < 0.999

    def test_matches_scipy_spearman(self):
        stats = pytest.importorskip("scipy.stats")
        fwd = forward_returns(self.tr, 1)
        sig = self.tr.shift(1)  # yesterday's return as signal
        ric = information_coefficient(sig, fwd, rank=True)
        d = ric.index[5]
        rho = stats.spearmanr(sig.loc[d], fwd.loc[d]).statistic
        assert ric.loc[d] == pytest.approx(rho)

    def test_min_obs(self):
        fwd = forward_returns(self.tr, 1)
        sig = fwd.copy()
        sig.iloc[:, 3:] = np.nan  # only 3 names
        assert information_coefficient(sig, fwd, min_obs=5).empty

    def test_ic_summary_and_analyzer(self):
        ic = pd.Series([0.1, 0.05, -0.02, 0.08])
        s = ic_summary(ic)
        assert s["icir"] == pytest.approx(ic.mean() / ic.std(ddof=1))
        assert s["hit_rate"] == 0.75
        assert np.isnan(ic_summary(pd.Series([0.1]))["icir"])
        table = SignalAnalyzer(self.tr).analyze(forward_returns(self.tr, 1), horizons=(1, 5))
        assert table.loc[("IC", 1), "mean"] == pytest.approx(1.0)
        assert table.loc[("RankIC", 5), "mean"] < 1.0
        assert len(SignalAnalyzer(self.tr).ic_series(self.tr.shift(1), 1, rank=True)) > 0

    def test_newey_west_with_no_lags_is_the_plain_t_stat(self):
        x = pd.Series(np.random.default_rng(5).normal(0.02, 0.1, 200))
        assert newey_west_tstat(x, 0) == pytest.approx(x.mean() / (x.std(ddof=0) / math.sqrt(len(x))))
        assert np.isnan(newey_west_tstat(pd.Series([0.1]), 3))
        assert np.isnan(newey_west_tstat(pd.Series([0.1, 0.1, 0.1]), 0))  # zero variance

    def test_overlapping_horizon_t_stat_is_adjusted(self):
        # a 5-day moving sum of iid noise mimics the overlap of 5-day forward windows
        e = pd.Series(np.random.default_rng(6).normal(0.01, 0.1, 600))
        ic = e.rolling(5).sum().dropna() / 5
        s5 = ic_summary(ic, horizon=5)
        assert abs(s5["t_stat"]) < abs(s5["t_stat_naive"]) * 0.7  # overlap inflates the naive figure
        assert s5["t_stat"] == pytest.approx(newey_west_tstat(ic, 4))
        s1 = ic_summary(ic, horizon=1)
        assert s1["t_stat"] == s1["t_stat_naive"]  # no overlap at horizon 1
