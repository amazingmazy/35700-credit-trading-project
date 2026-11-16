import math

import numpy as np
import pandas as pd
import pytest

from src.backtester import Backtester
from src.config import BacktestConfig
from src.performance import (PerformanceAnalyzer, drawdowns, equity_curve, information_ratio, max_drawdown,
                             sharpe_ratio)
from src.signal_analysis import SignalAnalyzer, forward_returns, ic_summary, information_coefficient
from src.strategies import EqualWeightLong

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
