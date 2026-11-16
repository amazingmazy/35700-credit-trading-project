"""CLI / plotting tests on synthetic parquet files, plus a slow full run on the real supplied data."""
from pathlib import Path

import numpy as np
import pytest

from main import main
from src.backtester import Backtester
from src.config import BacktestConfig
from src.data_loader import DataLoader
from src.plotting import plot_result
from src.strategies import EqualWeightLong, NoTrade
from tests.conftest import TRADING_DATES, corp_event, make_frames, vectorized_price_plus_dividend_pnl

REAL_DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture
def data_dir(tmp_path):
    eod, snap, corp, sym = make_frames(corp_rows=[corp_event("A", TRADING_DATES[3], 0.5)])
    eod.to_parquet(tmp_path / "etf_data_eod.parquet")
    snap.to_parquet(tmp_path / "etf_data_3pm.parquet")
    corp.to_parquet(tmp_path / "etf_dividends_splits.parquet")
    sym.to_parquet(tmp_path / "etf_symbology.parquet")
    return tmp_path


class TestCLI:
    def test_main_writes_outputs(self, data_dir, tmp_path):
        out = tmp_path / "out"
        m = main(["--data", str(data_dir), "--out", str(out), "--strategy", "momentum_ls", "--benchmark", "AAA"])
        assert "sharpe" in m and "information_ratio" in m
        for f in ("daily.csv", "metrics.json", "report.png"):
            assert (out / "momentum_ls" / f).exists()

    def test_main_list(self, capsys):
        assert main(["--list"]) == {}
        assert "equal_weight_long" in capsys.readouterr().out

    def test_plot_returns_figure(self, data_dir):
        md = DataLoader(data_dir).load()
        fig = plot_result(Backtester(md, NoTrade()).run())
        assert len(fig.axes) == 4


@pytest.mark.slow
@pytest.mark.skipif(not (REAL_DATA / "etf_data_eod.parquet").exists(), reason="supplied data not present")
class TestRealDataSmoke:
    @pytest.fixture(scope="class")
    def market(self):
        return DataLoader(REAL_DATA).load()

    def test_calendar(self, market):
        assert len(market.symbols) == 213
        eod_only = {"2020-10-12", "2020-11-11", "2021-10-11", "2025-10-13", "2025-11-11", "2026-10-12"}
        assert not set(market.dates.strftime("%Y-%m-%d")) & eod_only

    def test_known_split_mapped(self, market):
        pref = market.symbol_for("PREF")
        assert market.split_ratio[pref].max() == pytest.approx(5.0)
        # split-adjusted total return on the split day is small, raw return is ~ -80%
        d = market.split_ratio[pref].idxmax()
        assert abs(market.total_return.loc[d, pref]) < 0.05

    def test_equal_weight_full_run_invariants(self, market):
        r = Backtester(market, EqualWeightLong(), BacktestConfig(), record_orders=False).run()
        d = r.daily
        assert len(d) == len(market.dates)
        assert not d["gmv_breach"].any() and d["gmv"].max() <= 60e6
        assert d["pnl"].sum() == pytest.approx(d["equity"].iloc[-1] - 10e6, abs=1e-3)
        np.testing.assert_allclose(d["pnl"], d["price_pnl"] + d["dividends"] - d["costs"], atol=1e-4)
        assert d["dividends"].sum() > 0 and d["dv01"].iloc[-1] > 0
        alt = vectorized_price_plus_dividend_pnl(market, r.positions)
        np.testing.assert_allclose(d["price_pnl"] + d["dividends"], alt, atol=1e-4)
