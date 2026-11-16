"""Daily event loop tying data, strategy, risk, execution and portfolio together."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import BacktestConfig
from src.context import DecisionContext
from src.data_loader import MarketData
from src.execution import ExecutionHandler
from src.portfolio import Portfolio
from src.risk import RiskManager
from src.strategies.base import Strategy

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    strategy_name: str
    config: BacktestConfig
    daily: pd.DataFrame  # one row per trading date
    positions: pd.DataFrame  # shares held after the close, dates x symbols
    orders: pd.DataFrame  # order log
    signals: pd.DataFrame  # strategy.signal() per date (may be empty)
    targets: pd.DataFrame = field(default_factory=pd.DataFrame)  # post-risk targets

    @property
    def pnl(self) -> pd.Series:
        return self.daily["pnl"]

    def metrics(self, benchmark_returns: pd.Series | None = None, bill_rate: pd.Series | None = None) -> dict:
        from src.performance import PerformanceAnalyzer

        return PerformanceAnalyzer(self.config).summary(self, benchmark_returns, bill_rate)


class Backtester:
    def __init__(
        self,
        market: MarketData,
        strategy: Strategy,
        config: BacktestConfig | None = None,
        record_orders: bool = True,
        record_targets: bool = False,
    ):
        self.market = market
        self.strategy = strategy
        self.config = config or BacktestConfig()
        self.risk = RiskManager(self.config)
        self.execution = ExecutionHandler(self.config, record_orders=record_orders)
        self.record_targets = record_targets

    def run(self) -> BacktestResult:
        m, cfg = self.market, self.config
        symbols = m.symbols
        portfolio = Portfolio(symbols, cfg)

        close = m.eod["close"].to_numpy()
        volume = m.eod["volume"]
        dv01 = m.eod["dv01"].ffill().to_numpy() if "dv01" in m.eod else np.zeros_like(close)
        split = m.split_ratio.to_numpy()
        cash_div = m.cash_dividend.to_numpy()
        price_3pm = m.snap["price"]

        records, pos_rows, orders, sig_rows, tgt_rows = [], [], [], {}, {}
        warmup = int(getattr(self.strategy, "warmup", 0))
        # Data-quality exclusions (missing dividend records / unrecorded splits) are never held.
        blocked = m.excluded_mask() if cfg.clean_universe_only else None

        for i, date in enumerate(m.dates):
            portfolio.apply_corporate_actions(split[i], cash_div[i])
            positions = portfolio.positions
            ctx = DecisionContext(m, i, positions, portfolio.equity + portfolio.day_dividends, cfg)

            in_warmup = i < warmup
            raw = None if in_warmup else self.strategy.generate_targets(ctx)
            sig = None if in_warmup else self.strategy.signal(ctx)
            if sig is not None and len(sig):
                sig_rows[date] = sig

            risk = self.risk.check(raw, positions, price_3pm.iloc[i], blocked)
            order_qty = self.risk.orders_from_targets(risk.targets, positions)
            report = self.execution.execute(
                order_qty, date, m.eod["close"].iloc[i], volume.iloc[i], risk.rejected, risk.blocked
            )
            portfolio.apply_fills(report.fills.to_numpy(), close[i], report.costs.to_numpy())
            rec = portfolio.mark(close[i])

            gmv = portfolio.gmv()
            rec.update(
                date=date,
                gmv=gmv,
                long_mv=portfolio.long_mv(),
                short_mv=portfolio.short_mv(),
                dv01=portfolio.dv01(dv01[i]),
                n_positions=int(np.count_nonzero(portfolio.q)),
                n_orders=int(np.count_nonzero(order_qty.to_numpy())),
                n_blocked=len(risk.blocked),
                gross_scale=risk.gross_scale,
                worst_case_scale=risk.worst_case_scale,
                gmv_breach=gmv > cfg.gmv_cap + 1e-6,
                warmup=in_warmup,
            )
            records.append(rec)
            pos_rows.append(portfolio.q.copy())
            orders.extend(o.to_dict() for o in report.orders)
            if self.record_targets:
                tgt_rows[date] = risk.targets

        daily = pd.DataFrame(records).set_index("date")
        daily["return"] = daily["pnl"] / cfg.initial_capital
        daily["turnover"] = daily["traded_notional"] / cfg.initial_capital
        if daily["gmv_breach"].any():
            logger.warning("GMV cap breached on %d day(s)", int(daily["gmv_breach"].sum()))
        positions = pd.DataFrame(pos_rows, index=m.dates, columns=symbols)
        return BacktestResult(
            strategy_name=getattr(self.strategy, "name", type(self.strategy).__name__),
            config=cfg,
            daily=daily,
            positions=positions,
            orders=pd.DataFrame(orders),
            signals=pd.DataFrame(sig_rows).T.reindex(columns=symbols) if sig_rows else pd.DataFrame(),
            targets=pd.DataFrame(tgt_rows).T if tgt_rows else pd.DataFrame(),
        )
