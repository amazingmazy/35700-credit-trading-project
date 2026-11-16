"""Data-quality rules: missing dividend records, unrecorded splits, the clean universe and analytics cleaning."""
import numpy as np
import pandas as pd
import pytest

from src import data_quality as dq
from src.backtester import Backtester
from src.config import BacktestConfig
from src.context import DecisionContext
from src.data_loader import MarketData
from src.risk import RiskManager
from tests.conftest import ALL_DATES, TRADING_DATES, FixedTargets, base_prices, make_frames, make_market

SYMS = [f"S{k:02d}" for k in range(12)]


def monthly_cash(n_months=30, gap=(20, 21, 22), keep=("S00", "S01"), never=("S11",), stop_one=None):
    """Daily cash panel: every symbol pays 0.1 on the first business day of each month, except
    `never` (never pays), and in the `gap` months only `keep` have a record."""
    days = pd.bdate_range("2023-01-02", periods=n_months * 21)
    months = days.to_period("M")
    uniq = months.unique()
    cash = pd.DataFrame(0.0, index=days, columns=SYMS)
    for k, m in enumerate(uniq):
        first = days[months == m][0]
        payers = [s for s in SYMS if s not in never]
        if k in gap:
            payers = [s for s in payers if s in keep]
        if stop_one and k >= stop_one[1]:
            payers = [s for s in payers if s != stop_one[0]]
        cash.loc[first, payers] = 0.1
    return cash, uniq


class TestDividendGaps:
    def test_gap_months_and_missing_symbols(self):
        cash, uniq = monthly_cash()
        payers = dq.monthly_payers(cash)
        gaps = dq.gap_months(payers)
        assert list(gaps) == [uniq[20], uniq[21], uniq[22]]
        missing = dq.missing_dividend_months(payers, gaps)
        flagged = set(missing.columns[missing.any(axis=0)])
        assert flagged == set(SYMS) - {"S00", "S01", "S11"}  # complete funds and a never-payer are not flagged

    def test_one_fund_stopping_is_not_a_file_gap(self):
        # a single low-yield fund that stops paying must not create gap months
        cash, _ = monthly_cash(gap=(), stop_one=("S05", 14))
        assert len(dq.gap_months(dq.monthly_payers(cash))) == 0

    def test_final_month_is_skipped(self):
        cash, uniq = monthly_cash(gap=(29,))  # the shortfall is in the last month of the sample
        assert len(dq.gap_months(dq.monthly_payers(cash))) == 0

    def test_first_year_has_no_reference(self):
        cash, _ = monthly_cash(n_months=10, gap=(5,))
        assert len(dq.gap_months(dq.monthly_payers(cash))) == 0

    def test_monthly_payers_index_is_complete(self):
        cash, uniq = monthly_cash(n_months=6, gap=())
        p = dq.monthly_payers(cash)
        assert list(p.index) == list(uniq) and p["S00"].all() and not p["S11"].any()


class TestUnrecordedSplits:
    def setup_method(self):
        idx = pd.bdate_range("2024-01-02", periods=6)
        self.close = pd.DataFrame({
            "REV": [25.0, 25.1, 50.2, 50.3, 50.1, 50.2],  # 1-for-2 reverse split, not recorded
            "FWD": [100.0, 100.0, 50.0, 50.0, 50.0, 50.0],  # 2-for-1 split, recorded
            "DROP": [20.0, 20.0, 17.0, 17.0, 17.0, 17.0],  # a real -15% move
            "GAP": [30.0, np.nan, 15.0, 15.0, 15.0, 15.0],  # halves across a missing close, not recorded
        }, index=idx)
        self.split = pd.DataFrame(1.0, index=idx, columns=self.close.columns)
        self.split.loc[idx[2], "FWD"] = 2.0

    def test_flags_only_split_sized_jumps_without_a_record(self):
        out = dq.unrecorded_splits(self.close, self.split)
        assert set(out["symbol"]) == {"REV", "GAP"}
        rev = out[out["symbol"] == "REV"].iloc[0]
        assert rev["date"] == self.close.index[2] and rev["price_ratio"] == pytest.approx(2.0)

    def test_assess_builds_exclusion_list(self):
        cash = pd.DataFrame(0.0, index=self.close.index, columns=self.close.columns)
        rep = dq.assess(cash, self.close, self.split)
        assert set(rep.excluded) == {"REV", "GAP"} and set(rep.clean) == {"FWD", "DROP"}
        assert list(rep.excluded_mask) == [True, False, False, True]
        s = rep.summary()
        assert s["n_clean"] == 2 and s["n_excluded"] == 2 and len(s["unrecorded_splits"]) == 2


class TestCleanUniverseInEngine:
    def market_with_unrecorded_split(self):
        p = base_prices()
        p["B"][list(ALL_DATES).index(TRADING_DATES[5]):] = 25.0
        return make_market(prices=p)

    def test_synthetic_default_market_is_fully_clean(self, market):
        assert len(market.quality.excluded) == 0 and not market.excluded_mask().any()

    def test_universe_leaves_out_excluded_symbols(self):
        m = self.market_with_unrecorded_split()
        pos = pd.Series(0.0, index=m.symbols)
        assert list(DecisionContext(m, 3, pos, 10e6, BacktestConfig()).universe(1)) == ["A", "C"]
        cfg_all = BacktestConfig(clean_universe_only=False)
        assert list(DecisionContext(m, 3, pos, 10e6, cfg_all).universe(1)) == ["A", "B", "C"]

    def test_risk_check_blocks_targets(self):
        idx = pd.Index(["A", "B", "C"])
        pos = pd.Series([0.0, 0.0, 0.0], index=idx)
        px = pd.Series([10.0, 10.0, 10.0], index=idx)
        r = RiskManager(BacktestConfig()).check(pd.Series({"A": 100.0, "B": 50.0}), pos, px, np.array([False, True, False]))
        assert r.targets["B"] == 0 and r.targets["A"] == 100 and r.blocked == ["B"] and r.rejected == []
        r2 = RiskManager(BacktestConfig()).check(pd.Series({"A": 100.0}), pos, px, np.array([False, True, False]))
        assert r2.blocked == []  # nothing was requested in the blocked symbol

    def test_flag_off_restores_scaffold_behaviour(self):
        m = self.market_with_unrecorded_split()
        r = Backtester(m, FixedTargets({"B": 1000}), BacktestConfig(clean_universe_only=False)).run()
        assert r.positions["B"].iloc[0] == 1000 and r.daily["n_blocked"].sum() == 0

    def test_subset_keeps_full_sample_report(self):
        m = self.market_with_unrecorded_split()
        sub = m.subset(TRADING_DATES[7], TRADING_DATES[10])  # window after the jump
        assert list(sub.quality.excluded) == ["B"] and sub.excluded_mask().tolist() == [False, True, False]

    def test_market_without_report_excludes_nothing(self, market):
        bare = MarketData(market.dates, market.symbols, market.eod, market.snap, market.corp_actions,
                          market.split_ratio, market.cash_dividend, market.total_return, market.symbology)
        assert bare.quality is None and not bare.excluded_mask().any()


class TestReconcileAndCleaning:
    def test_reconcile_distributions(self):
        idx = pd.bdate_range("2024-01-01", "2025-12-31")
        close = pd.DataFrame(100.0, index=idx, columns=["A"])
        cash = pd.DataFrame(0.0, index=idx, columns=["A"])
        first_of_month = idx.to_series().groupby(idx.to_period("M")).first()
        cash.loc[first_of_month[first_of_month.dt.year == 2024], "A"] = 0.4  # 4.8% in 2024, nothing in 2025
        yld = pd.DataFrame(4.8, index=idx, columns=["A"])
        out = dq.reconcile_distributions(cash, close, yld).set_index("year")
        assert out.loc[2024, "dist_yield_pct"] == pytest.approx(4.8) and out.loc[2024, "shortfall_pct"] == pytest.approx(0.0)
        assert out.loc[2025, "shortfall_pct"] == pytest.approx(4.8)

    def test_clean_analytics_masks_whole_row(self):
        idx = pd.bdate_range("2024-01-01", periods=3)
        mk = lambda vals: pd.DataFrame({"A": vals, "B": [1.0, 1.0, 1.0]}, index=idx)
        eod = {"yield": mk([5.0, 12602.0, 5.0]), "duration": mk([4.0, 4.0, 13703.0]),
               "gspread": mk([100.0, 100.0, 100.0]), "spread": mk([100.0, 100.0, 100.0]), "close": mk([50.0, 50.0, 50.0])}
        out = dq.clean_analytics(eod)
        assert "close" not in out
        assert out["gspread"]["A"].isna().tolist() == [False, True, True]  # bad yield or duration blanks the row
        assert out["yield"]["B"].notna().all()
        eod["gspread"].iloc[0, 0] = -36000.0
        assert dq.clean_analytics(eod)["yield"]["A"].isna().all()
        assert dq.clean_analytics({"close": mk([1.0, 1.0, 1.0])}) == {}

    def test_despike_uses_past_data_only(self):
        idx = pd.bdate_range("2024-01-01", periods=40)
        s = pd.DataFrame({"A": np.full(40, 100.0)}, index=idx)
        s.iloc[30, 0] = 400.0  # isolated spike
        out = dq.despike(s)
        assert np.isnan(out.iloc[30, 0]) and out.drop(index=idx[30]).notna().all().all()
        # changing a later value cannot change an earlier decision
        s2 = s.copy()
        s2.iloc[35, 0] = 9999.0
        pd.testing.assert_series_equal(dq.despike(s2)["A"].iloc[:35], out["A"].iloc[:35])
