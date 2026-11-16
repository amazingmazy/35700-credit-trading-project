import numpy as np
import pandas as pd
import pytest

from src.data_loader import DataLoader, MarketData, classify_corp_action
from tests.conftest import ALL_DATES, HOLIDAY, TRADING_DATES, base_prices, corp_event, make_frames, make_market


class TestCalendarAndUniverse:
    def test_holiday_only_in_eod_is_removed(self, market):
        assert HOLIDAY not in market.dates
        assert market.dates.equals(TRADING_DATES)

    def test_dates_sorted_unique(self, market):
        assert market.dates.is_monotonic_increasing and market.dates.is_unique

    def test_only_universe_symbols(self, market):
        assert list(market.symbols) == ["A", "B", "C"]
        assert "X" not in market.eod["close"].columns

    def test_panels_aligned(self, market):
        for panel in list(market.eod.values()) + list(market.snap.values()):
            assert panel.index.equals(market.dates)
            assert panel.columns.equals(market.symbols)

    def test_ticker_mapping(self, market):
        assert market.symbol_for("AAA") == "A"
        assert market.ticker_for("B") == "BBB"
        assert market.tickers()["C"] == "CCC"
        with pytest.raises(KeyError):
            market.symbol_for("ZZZ")

    def test_subset(self, market):
        sub = market.subset(TRADING_DATES[2], TRADING_DATES[5])
        assert len(sub.dates) == 4
        assert sub.eod["close"].index.equals(sub.dates)


class TestSnapshotPrice:
    def test_price_uses_last(self, market):
        d = TRADING_DATES[0]
        assert market.snap["price"].loc[d, "A"] == pytest.approx(100.0 - 0.05)

    def test_missing_last_falls_back_to_mid(self):
        d = TRADING_DATES[3]
        m = make_market(snap_last_nan=[("A", d)])
        close = m.eod["close"].loc[d, "A"]
        assert m.snap["price"].loc[d, "A"] == pytest.approx(close - 0.05)  # mid of (c-.06, c-.04)
        assert np.isnan(m.snap["last"].loc[d, "A"])


class TestCorporateActions:
    def test_classification(self):
        assert classify_corp_action("Income") == "cash"
        assert classify_corp_action("Long Term Cap Gain") == "cash"
        assert classify_corp_action("Stock Split") == "split"
        assert classify_corp_action("Omitted") == "ignored"
        assert classify_corp_action("Discontinued") == "ignored"

    def test_cash_and_split_panels(self):
        d1, d2 = TRADING_DATES[2], TRADING_DATES[4]
        m = make_market(corp_rows=[corp_event("A", d1, 0.5), corp_event("B", d2, 2.0, "Stock Split"),
                                   corp_event("C", d1, 9.9, "Omitted")])
        assert m.cash_dividend.loc[d1, "A"] == 0.5
        assert m.cash_dividend.loc[d1, "C"] == 0.0  # ignored type
        assert m.split_ratio.loc[d2, "B"] == 2.0
        assert m.cash_dividend.drop(index=d1).to_numpy().sum() == 0.0

    def test_ex_date_on_removed_holiday_maps_to_next_trading_day(self):
        m = make_market(corp_rows=[corp_event("A", HOLIDAY, 0.3)])
        nxt = TRADING_DATES[TRADING_DATES > HOLIDAY][0]
        assert m.cash_dividend.loc[nxt, "A"] == 0.3

    def test_exact_duplicates_dropped(self):
        d = TRADING_DATES[2]
        m = make_market(corp_rows=[corp_event("A", d, 0.5), corp_event("A", d, 0.5)])
        assert m.cash_dividend.loc[d, "A"] == 0.5

    def test_total_return_includes_dividend_and_split(self):
        d = TRADING_DATES[4]
        m = make_market(corp_rows=[corp_event("B", d, 1.0)])
        i = m.dates.get_loc(d)
        assert m.total_return["B"].iloc[i] == pytest.approx(1.0 / 50.0)
        # split: price halves but total return is zero
        p = base_prices()
        k = list(ALL_DATES).index(d)
        p["B"][k:] = 25.0
        m2 = make_market(prices=p, corp_rows=[corp_event("B", d, 2.0, "Stock Split")])
        assert m2.total_return["B"].iloc[i] == pytest.approx(0.0)


class TestRiskFieldCleaning:
    def test_absurd_duration_and_negative_dv01_set_nan(self):
        eod, snap, corp, sym = make_frames()
        eod.loc[(eod.symbol == "A") & (eod.date == TRADING_DATES[1]), "duration"] = 13703.0
        eod.loc[(eod.symbol == "B") & (eod.date == TRADING_DATES[1]), "dv01"] = -175.0
        m = MarketData.from_frames(eod, snap, corp, sym)
        assert np.isnan(m.eod["duration"].loc[TRADING_DATES[1], "A"])
        assert np.isnan(m.eod["dv01"].loc[TRADING_DATES[1], "A"])
        assert np.isnan(m.eod["dv01"].loc[TRADING_DATES[1], "B"])
        assert m.eod["dv01"].loc[TRADING_DATES[1], "C"] > 0


class TestDataLoaderIO:
    def test_load_from_parquet(self, tmp_path):
        eod, snap, corp, sym = make_frames(corp_rows=[corp_event("A", TRADING_DATES[2], 0.5)])
        eod.to_parquet(tmp_path / "etf_data_eod.parquet")
        snap.to_parquet(tmp_path / "etf_data_3pm.parquet")
        corp.to_parquet(tmp_path / "etf_dividends_splits.parquet")
        sym.to_parquet(tmp_path / "etf_symbology.parquet")
        m = DataLoader(tmp_path).load(start=TRADING_DATES[1])
        assert m.dates[0] == TRADING_DATES[1]
        assert m.cash_dividend.loc[TRADING_DATES[2], "A"] == 0.5

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            DataLoader(tmp_path).load()
