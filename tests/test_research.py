"""Research library: buckets, panels and rates hedge, statistics, hazard curve, Merton, and signal timing."""
import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from src.research import hazard, merton, signals, stats
from src.research.panels import Panels, build_panels, hedged_return, rates_return
from src.research.universe import BUCKET_OF, CREDIT_BUCKETS, HEDGE_ANCHORS, IG_MATURITY, bucket_of, in_bucket
from tests.conftest import ALL_DATES, TRADING_DATES, base_prices, make_market

DUR = dict(SHY=1.9, IEI=4.4, IEF=7.5, TLH=13.0, TLT=17.0, LQD=8.5, VCIT=6.2, VCSH=2.7, VCLT=13.0, IGIB=6.1,
           IGSB=2.8, SPIB=4.2, USIG=6.9, HYG=3.4, JNK=3.5, USHY=3.6, SJNK=2.4, BKLN=3.6)
TICKERS = list(DUR)
CREDIT_BETA = dict(LQD=1.0, VCIT=0.8, VCSH=0.3, VCLT=1.4, IGIB=0.8, IGSB=0.3, SPIB=0.5, USIG=0.9,
                   HYG=1.0, JNK=1.05, USHY=1.0, SJNK=0.7, BKLN=0.5)


def make_panels(n=260, seed=0) -> Panels:
    """Synthetic panels: one parallel rates factor, one credit factor, idiosyncratic noise."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n)
    dy, cs = rng.normal(0, 5e-4, n), rng.normal(0, 1e-3, n)
    frame = lambda f: pd.DataFrame({t: f(t) for t in TICKERS}, index=idx)
    ret = frame(lambda t: -DUR[t] * dy + CREDIT_BETA.get(t, 0.0) * cs + (rng.normal(0, 2e-4, n) if t in CREDIT_BETA else 0.0))
    to3 = frame(lambda t: 0.8 * ret[t] + (rng.normal(0, 1e-4, n) if t in CREDIT_BETA else 0.0))
    close = 100.0 * (1.0 + ret).cumprod()
    price = close.shift(1) * (1.0 + to3)
    nav = close / (1.0 + frame(lambda t: rng.normal(0, 2e-3, n)))
    duration = frame(lambda t: np.full(n, DUR[t]))
    ttm = duration * 1.2
    level = 120.0 - 40.0 * (1 - np.exp(-ttm)) / ttm  # spread curve rising with maturity
    gspread = level + frame(lambda t: rng.normal(0, 3.0, n))
    gspread[[t for t in TICKERS if t not in CREDIT_BETA]] = 0.0  # Treasuries carry no spread
    return Panels(
        tickers=TICKERS, bucket=bucket_of(TICKERS), close=close, nav=nav,
        volume=frame(lambda t: np.full(n, 1e6)), price_3pm=price, mid_3pm=price.copy(),
        ret=ret, ret_to_3pm=to3, ret_to_3pm_mid=to3.copy(), ret_3pm_to_close=(1 + ret) / (1 + to3) - 1,
        analytics={"duration": duration, "avgttm": ttm, "gspread": gspread, "yield": 4.0 + gspread / 100.0},
    )


EOD_PANELS = ("close", "nav", "volume", "ret", "ret_3pm_to_close")
SNAP_PANELS = ("price_3pm", "mid_3pm", "ret_to_3pm", "ret_to_3pm_mid")


def poison_after(p: Panels, k: int) -> Panels:
    """Corrupt everything a day-k decision may not see: EOD rows >= k and 3PM rows > k."""
    q = make_panels()
    for name in EOD_PANELS:
        getattr(q, name).iloc[k:] = 7.7
    for name in SNAP_PANELS:
        getattr(q, name).iloc[k + 1:] = 7.7
    for a in q.analytics.values():
        a.iloc[k:] = 7.7
    return q


class TestUniverse:
    def test_buckets(self):
        assert BUCKET_OF["HYG"] == "HY" and BUCKET_OF["LQD"] == "IG" and BUCKET_OF["BKLN"] == "Loan"
        assert BUCKET_OF["EUHY"] == "HY-Global" and BUCKET_OF["TLT"] == "UST"
        assert bucket_of(["HYG", "ZZZ"]).tolist() == ["HY", "Other"]
        assert in_bucket(["TLT", "HYG", "LQD", "ZZZ"], "IG", "HY") == ["HYG", "LQD"]
        assert set(HEDGE_ANCHORS) <= {t for t, b in BUCKET_OF.items() if b == "UST"}
        assert "UST" not in CREDIT_BUCKETS

    def test_ig_maturity_partitions_the_ig_bucket(self):
        listed = [t for s in IG_MATURITY.values() for t in s.split()]
        assert sorted(listed) == sorted(t for t, b in BUCKET_OF.items() if b == "IG") and len(listed) == 40

    def test_no_ticker_in_two_buckets(self):
        assert len(BUCKET_OF) == 30 + 40 + 4 + 23 + 4 + 2 + 3


class TestPanels:
    def test_build_from_market(self, market):
        p = build_panels(market)
        assert p.tickers == ["AAA", "BBB", "CCC"] and list(p.close.columns) == p.tickers
        assert p.dates.equals(market.dates) and p.nav.isna().all().all()  # the synthetic market has no fnav
        d = TRADING_DATES[3]
        prev = market.eod["close"]["A"].shift(1)[d]
        assert p.ret_to_3pm.loc[d, "AAA"] == pytest.approx(market.snap["price"].loc[d, "A"] / prev - 1)
        assert p.ret_3pm_to_close.loc[d, "AAA"] == pytest.approx(market.eod["close"].loc[d, "A"] / market.snap["price"].loc[d, "A"] - 1)
        assert set(p.analytics) == {"duration", "dv01", "spread"} and p.a("duration").iloc[0, 0] == 5.0
        assert p.cap_usd().iloc[:20].isna().all().all()

    def test_clean_only_drops_excluded(self):
        prices = base_prices()
        prices["B"][list(ALL_DATES).index(TRADING_DATES[5]):] = 25.0  # unrecorded split
        m = make_market(prices=prices)
        assert build_panels(m).tickers == ["AAA", "CCC"]
        assert build_panels(m, clean_only=False, despike_spreads=False).tickers == ["AAA", "BBB", "CCC"]

    def test_rates_hedge_removes_a_parallel_move(self):
        idx = pd.bdate_range("2024-01-01", periods=6)
        dy = np.array([0.0, 1e-3, -2e-3, 5e-4, 0.0, 1e-3])
        dur = pd.DataFrame({"S": 2.0, "L": 10.0, "MID": 6.0, "XL": 20.0}, index=idx)
        ret = pd.DataFrame({c: -dur[c].to_numpy() * dy for c in dur}, index=idx)
        rr = rates_return(ret, dur, anchors=("S", "L"))
        assert rr.iloc[0].isna().all()  # the duration is lagged one row
        np.testing.assert_allclose(rr["MID"].iloc[1:], ret["MID"].iloc[1:], atol=1e-15)
        np.testing.assert_allclose(rr["XL"].iloc[1:], 20.0 * (ret["L"].iloc[1:] / 10.0), atol=1e-15)  # flat beyond the anchors
        np.testing.assert_allclose(hedged_return(ret, dur, anchors=("S", "L")).iloc[1:], 0.0, atol=1e-15)

    def test_hedge_skips_days_with_a_missing_anchor(self):
        p = make_panels(30)
        p.ret.loc[p.dates[10], "IEF"] = np.nan
        assert hedged_return(p.ret, p.a("duration")).loc[p.dates[10]].isna().all()
        assert hedged_return(p.ret, p.a("duration"))[list(HEDGE_ANCHORS)].iloc[11:].abs().max().max() < 1e-12


class TestStats:
    def test_trailing_z_excludes_the_current_row(self):
        x = pd.Series(np.arange(50, dtype=float))
        z = stats.trailing_z(x, window=10, min_periods=5)
        assert z.iloc[:5].isna().all()
        assert z.iloc[20] == pytest.approx((20 - np.arange(10, 20).mean()) / np.arange(10, 20).std(ddof=1))
        assert stats.trailing_z(pd.Series(np.ones(30)), 10, 5).isna().all()  # zero variance

    def test_half_life(self):
        rng = np.random.default_rng(0)
        x = np.zeros(5000)
        for i in range(1, 5000):
            x[i] = 0.5 * x[i - 1] + rng.normal()
        assert stats.half_life(pd.Series(x)) == pytest.approx(1.0, abs=0.1)
        assert np.isnan(stats.half_life(pd.Series(x[:50]))) and np.isnan(stats.half_life(pd.Series(rng.normal(size=500)).diff()))

    def test_forward_sum(self):
        r = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        assert stats.forward_sum(r, 2).tolist()[:3] == [5.0, 7.0, 9.0] and stats.forward_sum(r, 2).iloc[3:].isna().all()
        with pytest.raises(ValueError):
            stats.forward_sum(r, 0)

    def test_demean_by_group(self):
        x = pd.DataFrame([[1.0, 3.0, 10.0, 20.0]], columns=list("abcd"))
        assert stats.demean_by_group(x, None).iloc[0].tolist() == [-7.5, -5.5, 1.5, 11.5]
        g = pd.Series({"a": "g1", "b": "g1", "c": "g2", "d": "g2"})
        assert stats.demean_by_group(x, g).iloc[0].tolist() == [-1.0, 1.0, -5.0, 5.0]

    def test_rank_ic_and_table(self):
        rng = np.random.default_rng(1)
        idx = pd.bdate_range("2024-01-01", periods=80)
        ret = pd.DataFrame(rng.normal(0, 0.01, (80, 8)), index=idx, columns=list("abcdefgh"))
        perfect = stats.forward_sum(ret, 1)
        assert np.allclose(stats.rank_ic(perfect, stats.forward_sum(ret, 1)), 1.0)
        tab = stats.ic_table(perfect, ret, horizons=(1, 5))
        assert tab.loc[1, "mean_ic"] == pytest.approx(1.0) and 0 < tab.loc[5, "mean_ic"] < 1 and tab.loc[5, "n_days"] == 75
        # a signal that only separates the two groups has no IC once groups are demeaned
        g = pd.Series({c: ("x" if c in "abcd" else "y") for c in ret.columns})
        between = pd.DataFrame(np.tile([0, 0, 0, 0, 1, 1, 1, 1.0], (80, 1)), index=idx, columns=ret.columns)
        between += rng.normal(0, 1e-9, between.shape)
        shifted = ret.copy()
        shifted[list("efgh")] += 0.05
        assert stats.rank_ic(between, shifted.shift(-1)).mean() > 0.6
        assert abs(stats.rank_ic(between, shifted.shift(-1), groups=g).mean()) < 0.2
        assert stats.rank_ic(perfect.iloc[:, :3], perfect.iloc[:, :3], min_obs=5).empty

    def test_predict_ts(self):
        rng = np.random.default_rng(2)
        sig = pd.Series(rng.normal(size=600))
        ret = 0.01 * sig.shift(1) + pd.Series(rng.normal(0, 0.01, 600))  # signal(T) drives the return of T+1
        out1, out5 = stats.predict_ts(sig, ret, 1), stats.predict_ts(sig, ret, 5)
        assert out1["corr"] > 0.5 and out1["slope"] == pytest.approx(0.01, rel=0.2) and out1["t_stat"] > 10
        assert 0 < out5["corr"] < out1["corr"] and out5["n"] == 595
        assert np.isnan(stats.predict_ts(sig.iloc[:5], ret.iloc[:5])["t_stat"])
        assert np.isnan(stats.predict_ts(pd.Series(np.ones(50)), ret.iloc[:50])["slope"])

    def test_financing_adjusted_sharpe(self):
        rng = np.random.default_rng(3)
        idx = pd.bdate_range("2024-01-01", periods=300)
        r = pd.Series(0.05 / 252 + rng.normal(0, 1e-4, 300), index=idx)  # a fund that earns the bill rate
        bill = pd.Series(0.05, index=idx)
        assert stats.sharpe(r) > 5 and abs(stats.financing_adjusted_sharpe(r, 1.0, bill)) < 2
        assert stats.financing_adjusted_sharpe(r, 0.0, bill) == pytest.approx(stats.sharpe(r))
        assert stats.financing_adjusted_sharpe(r, pd.Series(1.0, index=idx), bill) == pytest.approx(stats.financing_adjusted_sharpe(r, 1.0, bill))
        assert np.isnan(stats.sharpe(pd.Series([0.01]))) and np.isnan(stats.sharpe(pd.Series([0.0, 0.0, 0.0])))


class TestHazard:
    def test_loadings(self):
        X = hazard.ns_loadings(np.array([0.5, 5.0, 30.0]), lam=2.0)
        assert X.shape == (3, 3) and np.allclose(X[:, 0], 1.0)
        assert X[0, 1] > X[1, 1] > X[2, 1] > 0  # the slope loading decays with maturity
        assert hazard.ns_loadings(np.array([1.0]), 2.0, n_terms=2).shape == (1, 2)
        with pytest.raises(ValueError):
            hazard.ns_loadings(np.array([1.0]), 2.0, n_terms=4)

    def setup_method(self):
        idx = pd.bdate_range("2024-01-01", periods=5)
        self.ttm = pd.DataFrame(np.tile([1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0], (5, 1)), index=idx, columns=list("abcdefgh"))
        self.theta = np.array([150.0, -60.0, 20.0])
        self.spread = pd.DataFrame(hazard.ns_loadings(self.ttm.iloc[0].to_numpy(), 2.0) @ self.theta, index=self.ttm.columns).T
        self.spread = pd.concat([self.spread] * 5).set_axis(idx)

    def test_fit_recovers_an_exact_curve_and_isolates_a_bump(self):
        fit = hazard.fit_curve_panel(self.spread, self.ttm, lam=2.0)
        np.testing.assert_allclose(fit.theta.iloc[0], self.theta, rtol=1e-8)
        assert fit.edges.abs().max().max() < 1e-8 and fit.rmse.max() < 1e-8
        bumped = self.spread.copy()
        bumped.loc[bumped.index[2], "d"] += 30.0
        e = hazard.fit_curve_panel(bumped, self.ttm, lam=2.0).edges.loc[bumped.index[2]]
        assert e["d"] > 15 and e["d"] == e.max() and abs(e.sum()) < 30

    def test_skips_thin_cross_sections_and_respects_weights(self):
        thin = self.spread.copy()
        thin.iloc[1, :4] = np.nan  # only 4 names left, fewer than n_terms + 3
        fit = hazard.fit_curve_panel(thin, self.ttm, lam=2.0)
        assert fit.edges.iloc[1].isna().all() and np.isnan(fit.rmse.iloc[1]) and fit.edges.iloc[0].notna().all()
        bumped = self.spread.copy()
        bumped["d"] += 30.0
        w = pd.DataFrame(1.0, index=bumped.index, columns=bumped.columns)
        w["d"] = 1e-9  # a name with no weight does not pull the curve
        assert hazard.fit_curve_panel(bumped, self.ttm, w, lam=2.0).edges["d"].iloc[0] == pytest.approx(30.0, abs=1e-4)

    def test_slope_matches_a_finite_difference(self):
        fit = hazard.fit_curve_panel(self.spread, self.ttm, lam=2.0)
        s = fit.slope(self.ttm, step=0.25)
        curve = lambda t: hazard.ns_loadings(np.array([t]), 2.0) @ self.theta
        assert s["d"].iloc[0] == pytest.approx(float((curve(5.0) - curve(4.75))[0] / 0.25))
        short = self.ttm.copy()
        short["a"] = 0.2  # shorter than the step
        assert hazard.fit_curve_panel(self.spread, self.ttm, lam=2.0).slope(short)["a"].isna().all()

    def test_credit_triangle(self):
        assert hazard.implied_hazard(300.0, 0.4) == pytest.approx(500.0)
        assert hazard.equalizing_recovery(330.0, 550.0) == pytest.approx(0.4)
        with pytest.raises(ValueError):
            hazard.implied_hazard(100.0, 1.0)
        gap = hazard.log_hazard_gap(pd.Series([300.0, -5.0]), pd.Series([300.0, 300.0]), 0.8, 0.4)
        assert gap.iloc[0] == pytest.approx(math.log(3.0)) and np.isnan(gap.iloc[1])


class TestMerton:
    A, K, SIG, R, T = 125.0, 100.0, 0.2, 0.04, 5.0  # the lecture-6 example

    def bs_call(self, A):
        d1 = (math.log(A / self.K) + (self.R + self.SIG**2 / 2) * self.T) / (self.SIG * math.sqrt(self.T))
        return A * norm.cdf(d1) - self.K * math.exp(-self.R * self.T) * norm.cdf(d1 - self.SIG * math.sqrt(self.T))

    def test_debt_is_assets_minus_call(self):
        b_over_a, spread, debt_vol = merton.merton_debt(self.K / self.A, self.SIG, self.T, self.R)
        B = self.A - self.bs_call(self.A)
        assert b_over_a * self.A == pytest.approx(B)
        assert spread == pytest.approx(-math.log(B / self.K) / self.T - self.R)
        h = 1e-4  # debt vol = (dB/dA)(A/B) sigma_A by finite difference
        dBdA = ((self.A + h) - self.bs_call(self.A + h) - (self.A - h) + self.bs_call(self.A - h)) / (2 * h)
        assert debt_vol == pytest.approx(dBdA * self.A / B * self.SIG, rel=1e-6)

    def test_spread_rises_with_leverage_and_vol(self):
        s = lambda L, v: merton.merton_debt(L, v, 5.0, 0.04)[1]
        assert s(0.9, 0.2) > s(0.6, 0.2) > 0 and s(0.8, 0.3) > s(0.8, 0.15)

    def test_solvers_round_trip(self):
        _, s, v = merton.merton_debt(1.3, 0.06, 4.0, 0.04)
        L, sig = merton.solve_leverage_vol(s, v, 4.0, 0.04)
        assert L == pytest.approx(1.3, rel=1e-5) and sig == pytest.approx(0.06, rel=1e-5)
        assert merton.implied_leverage(s, 0.06, 4.0, 0.04) == pytest.approx(1.3, rel=1e-6)
        assert np.isnan(merton.solve_leverage_vol(-0.01, v, 4.0, 0.04)[0]) and np.isnan(merton.solve_leverage_vol(np.nan, v, 4.0, 0.04)[0])
        assert np.isnan(merton.implied_leverage(np.nan, 0.06, 4.0, 0.04)) and np.isnan(merton.implied_leverage(5.0, 0.06, 4.0, 0.04))

    def test_panel_solvers(self):
        idx = pd.bdate_range("2024-01-01", periods=4)
        true = [(1.2, 0.05), (1.25, 0.06), (1.3, 0.07), (1.3, 0.07)]
        obs = [merton.merton_debt(L, v, 4.0, 0.04)[1:] for L, v in true]
        s = pd.Series([o[0] for o in obs], index=idx)
        v = pd.Series([o[1] for o in obs], index=idx)
        v.iloc[3] = np.nan
        T, r = pd.Series(4.0, index=idx), pd.Series(0.04, index=idx)
        out = merton.solve_panel(s, v, T, r)
        np.testing.assert_allclose(out.iloc[:3].to_numpy(), np.array(true[:3]), rtol=1e-5)
        assert out.iloc[3].isna().all()
        lev = merton.implied_leverage_panel(s, pd.Series([t[1] for t in true], index=idx), T, r)
        np.testing.assert_allclose(lev, [t[0] for t in true], rtol=1e-6)

    def test_option_vol_tends_to_the_instantaneous_debt_vol(self):
        _, _, v = merton.merton_debt(1.3, 0.06, 4.0, 0.04)
        assert merton.debt_option_implied_vol(1.3, 0.06, 4.0, 0.04, 1 / 252) == pytest.approx(v, rel=0.02)
        assert merton.debt_option_implied_vol(1.3, 0.06, 4.0, 0.04, 1.0) > 0
        assert np.isnan(merton.debt_option_implied_vol(1.3, 0.06, 4.0, 0.04, 0.0)) and np.isnan(merton.debt_option_implied_vol(np.nan, 0.06, 4.0, 0.04, 1.0))

    def test_rates_and_credit_vol_split(self):
        idx = pd.bdate_range("2024-01-01", periods=3)
        avol = pd.DataFrame({"S": [2.0, 2.0, np.nan], "L": [10.0, 10.0, 10.0]}, index=idx)  # 1% yield vol at both ends
        adur = pd.DataFrame({"S": 2.0, "L": 10.0}, index=idx)
        rv = merton.rates_vol(pd.Series(5.0, index=idx), avol, adur)
        assert rv.iloc[0] == pytest.approx(5.0) and np.isnan(rv.iloc[2])
        total = pd.Series([13.0, 4.0, 13.0], index=idx)
        c0 = merton.credit_only_vol(total, pd.Series(5.0, index=idx), 0.0)
        assert c0.iloc[0] == pytest.approx(12.0) and np.isnan(c0.iloc[1])  # 4 < 5: no real solution
        c = merton.credit_only_vol(total, pd.Series(5.0, index=idx), -0.2).iloc[0]
        assert 5.0**2 + c**2 + 2 * (-0.2) * 5.0 * c == pytest.approx(13.0**2)


class TestSignalTiming:
    """Every signal dated T must be unchanged when data a day-T decision cannot see is corrupted."""

    K = 200

    def check(self, fn):
        base = fn(make_panels())
        bad = fn(poison_after(make_panels(), self.K))
        pd.testing.assert_frame_equal(base.iloc[: self.K + 1], bad.iloc[: self.K + 1])
        assert base.iloc[self.K - 20: self.K + 1].notna().any().any()  # the signal is populated near T
        assert not base.iloc[self.K + 1:].equals(bad.iloc[self.K + 1:])  # and the corruption does bite later
        return base

    def test_nav_discount_lagged(self):
        s = self.check(signals.nav_discount_lagged)
        p = make_panels()
        b = signals.nav_basis_bp(p.close, p.nav)
        d = p.dates[150]
        hist = b["HYG"].iloc[150 - 61:150 - 1]  # the 60 closes before T-1
        assert s.loc[d, "HYG"] == pytest.approx(-(b["HYG"].iloc[149] - hist.mean()) / hist.std(ddof=1))

    def test_nav_discount_3pm(self):
        self.check(signals.nav_discount_3pm)

    def test_residual_3pm_reversal(self):
        s = self.check(signals.residual_3pm_reversal)
        assert "HYG" not in s and "LQD" not in s and "TLT" not in s and {"JNK", "VCIT", "BKLN"} <= set(s.columns)

    def test_hazard_edge_deviation(self):
        ig = in_bucket(TICKERS, "IG")
        self.check(lambda p: signals.edge_deviation_z(signals.hazard_edge_fit(p, ig).edges))

    def test_rolldown(self):
        ig = in_bucket(TICKERS, "IG")

        def rd(p):
            fit = signals.hazard_edge_fit(p, ig)
            return signals.rolldown_pct(fit, p.a("avgttm")[ig], p.a("duration")[ig], per_bp=True).shift(1)

        out = self.check(rd)
        # the synthetic spread curve slopes up, most steeply at the short end
        assert (out[["VCSH", "IGSB", "SPIB"]].iloc[100:].mean() > 0).all()


class TestSignalValues:
    def test_nav_basis(self):
        idx = pd.bdate_range("2024-01-01", periods=2)
        close = pd.DataFrame({"A": [100.5, 80.0]}, index=idx)
        nav = pd.DataFrame({"A": [100.0, 100.0]}, index=idx)
        b = signals.nav_basis_bp(close, nav)["A"]
        assert b.iloc[0] == pytest.approx(50.0) and np.isnan(b.iloc[1])  # -2000bp is treated as bad data

    def test_factor_residual_removes_rates_and_reference(self):
        p = make_panels(400, seed=4)
        beta = signals.reference_beta(p)
        assert beta["JNK"].iloc[-1] == pytest.approx(CREDIT_BETA["JNK"], rel=0.1)  # recovers the loading on HYG
        resid = signals.factor_residual(p, p.ret, beta)
        raw_sd, res_sd = p.ret["JNK"].std(), resid["JNK"].std()
        assert res_sd < 0.3 * raw_sd  # rates and credit factors explain most of the move

    def test_edge_signal_ranks_a_cheap_fund_first(self):
        p = make_panels()
        ig = in_bucket(TICKERS, "IG")
        p.analytics["gspread"].loc[p.dates[220]:, "VCIT"] += 40.0  # VCIT cheapens against the curve
        z = signals.edge_deviation_z(signals.hazard_edge_fit(p, ig).edges)
        assert z.loc[p.dates[222]].idxmax() == "VCIT" and z.loc[p.dates[220], "VCIT"] < 3  # known a day later

    def test_liquidity_weights(self):
        p = make_panels()
        w = signals.liquidity_weights(p, ["VCSH", "VCLT"]).iloc[-1]
        assert w["VCSH"] > w["VCLT"] > 0  # same volume, shorter duration gets more weight
