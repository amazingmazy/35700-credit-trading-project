"""Run a registered strategy on the supplied data and write results.

Usage:
    python main.py --strategy equal_weight_long
    python main.py --strategy momentum_ls --start 2021-01-01 --out results
    python main.py --list
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.backtester import Backtester
from src.config import BacktestConfig
from src.data_loader import DataLoader
from src.performance import PerformanceAnalyzer
from src.plotting import plot_result
from src.signal_analysis import SignalAnalyzer
from src.strategies import STRATEGY_REGISTRY


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--strategy", default="equal_weight_long")
    p.add_argument("--data", default="data")
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--out", default="results")
    p.add_argument("--benchmark", default="AGG", help="ticker for IR (use 'none' to skip)")
    p.add_argument("--list", action="store_true", help="list registered strategies")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.list:
        print("\n".join(sorted(STRATEGY_REGISTRY)))
        return {}

    config = BacktestConfig()
    market = DataLoader(args.data).load(args.start, args.end)
    strategy = STRATEGY_REGISTRY[args.strategy]()
    result = Backtester(market, strategy, config).run()

    bench = None
    if args.benchmark and args.benchmark.lower() != "none":
        bench = market.total_return[market.symbol_for(args.benchmark)].fillna(0.0)
    metrics = PerformanceAnalyzer(config).summary(result, bench)

    print(f"\n=== {result.strategy_name} ({market.dates[0].date()} -> {market.dates[-1].date()}) ===")
    print(PerformanceAnalyzer.format_summary(metrics))

    out = Path(args.out) / result.strategy_name
    out.mkdir(parents=True, exist_ok=True)
    result.daily.to_csv(out / "daily.csv")
    result.positions.loc[:, (result.positions != 0).any()].to_csv(out / "positions.csv")
    if not result.orders.empty:
        result.orders.to_csv(out / "orders.csv", index=False)
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    if not result.signals.empty:
        ic = SignalAnalyzer(market.total_return, config.periods_per_year).analyze(result.signals)
        print("\nSignal diagnostics:\n", ic.round(4).to_string())
        ic.to_csv(out / "signal_ic.csv")
    if not args.no_plot:
        plot_result(result, str(out / "report.png"))
    print(f"\nOutputs written to {out}")
    return metrics


if __name__ == "__main__":
    main()
