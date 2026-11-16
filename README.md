# Credit ETF Backtester — FINM 35700 Group Project

Modular, object-oriented daily backtester for the Credit ETF trading strategy project.
New strategies plug in as one class. The engine enforces the project's information, execution and capital rules, and pytest checks them.

## Run

The supplied data is not in the repo. Copy the four course parquet files (`etf_symbology`, `etf_data_eod`,
`etf_data_3pm`, `etf_dividends_splits`) into `data/`. Real-data tests skip automatically without them.

```bash
# Python 3.12 (the `finm` conda env works); base Anaconda on this machine has a broken numpy
conda activate finm            # or call C:\Users\<you>\anaconda3\envs\finm\python.exe directly
pip install -r requirements.txt

python main.py --list                              # registered strategies
python main.py --strategy equal_weight_long        # full sample, writes results/<strategy>/
python main.py --strategy momentum_ls --start 2021-01-01 --benchmark AGG

pytest -m "not slow"                               # fast synthetic tests (~30s)
pytest                                             # + full real-data smoke run (~1 min)
coverage run --source=src -m pytest && coverage report -m
```

Outputs per run: `daily.csv` (PnL, dividends, costs, equity, GMV, DV01, turnover…), `positions.csv`,
`orders.csv`, `metrics.json`, `signal_ic.csv` (if the strategy defines `signal`), `report.png`.

## Adding a strategy

```python
# src/strategies/my_strategy.py   (then import it in src/strategies/__init__.py)
import pandas as pd
from src.strategies.base import WeightStrategy, register

@register("carry")
class Carry(WeightStrategy):
    warmup = 20

    def signal(self, ctx):                       # optional: enables IC / Rank IC / ICIR / decay
        return ctx.eod_row("spread", lag=1).reindex(ctx.universe(20))

    def target_weights(self, ctx):               # fractions of config.gmv_target; sum |w| <= 1
        s = self.signal(ctx).dropna()
        z = (s - s.mean()) / s.std()
        return z / z.abs().sum()
```

Use `Strategy.generate_targets(ctx) -> pd.Series` (target **shares**) for full control, or
`WeightStrategy.target_weights(ctx)` (fractions of the $57MM gross target). In the returned Series, omitted symbols mean
*flatten* and NaN means *hold*. A held position can still be scaled down by the GMV checks.

## What a strategy can see (`DecisionContext`)

| Method | Data | Rule |
|---|---|---|
| `history(field, lookback)` / `eod_row(field, lag>=1)` | EOD fields + `total_return` | dates **< T** only |
| `snapshot()` / `price` | 3PM last/bid/ask/mid; decision price = last, else mid | day T |
| `known_corp_actions()` | events with `declared_date <= T` | known at submission |
| `positions`, `equity`, `universe(min_history)`, `weights_to_shares()` | portfolio state and helpers | point-in-time |

Day-T close and volume are only used by `ExecutionHandler`/`Portfolio`. `tests/test_context.py` corrupts every value from day T onward
and asserts the day-T decisions do not change.

## Architecture

```
DataLoader ─► MarketData (dates × symbols panels, corp actions, total returns)
                   │
Backtester loop, each date T:
  Portfolio.apply_corporate_actions(T)    splits rescale shares; cash distributions to holders (shorts pay)
  DecisionContext(T) ─► Strategy.generate_targets / signal
  RiskManager.check                       normalise, untradable → hold, scale to GMV target, worst-case GMV
  ExecutionHandler.execute                MOC @ close_T, ≤ floor(3% · volume_T), $0.0035/share, remainder cancelled
  Portfolio.apply_fills / mark            mark-to-market, PnL, GMV, DV01
                   ▼
BacktestResult ─► PerformanceAnalyzer (required metrics, IR vs AGG, costs/turnover)
              └─► SignalAnalyzer (IC, Rank IC, ICIR, decay at 1/5/20 days)
```

## Assumptions (to state in the presentation)

- **Calendar:** dates common to the EOD and 3PM files. This removes the 6 EOD-only bond holidays (Columbus Day and Veterans Day).
- **Universe:** the 213 symbology ISINs. Strategies should filter point-in-time with `ctx.universe(min_history)`. 9 ETFs list during the sample and none delist.
- **Corporate actions** are applied on `ex_date_adj`; an ex-date on a removed holiday rolls to the next trading day.
  - Income, Cap Gains, Special/Regular Cash and ROC are paid as cash per share held at the T-1 close (short positions pay them).
  - A stock split multiplies the share count by `dividend_amount`.
  - Omitted, Cancelled and Discontinued are ignored. Exact duplicate rows are dropped.
- **Prices** are raw closes. Missing closes are marked at the last valid close, and no trade can happen without a close.
- **Execution:** whole shares (truncated toward zero). Fill = sign × min(|order|, floor(0.03 × volume_T)). Zero or NaN volume → no fill. Unfilled remainder is cancelled, and the strategy re-targets the next day.
- **Gross exposure:** sized to $57MM at 3PM prices (a $3MM buffer for 3PM→close moves). By default the sizing assumes the worst case: gross-reducing orders may not fill while gross-increasing orders fill fully. Post-trade GMV is checked against the $60MM cap and any breach is flagged.
- **DV01:** per share = `dv01 / 100` (YAS_RISK ≈ close × modDur / 100), so aggregate DV01 = Σ q · dv01 / 100 in $ per −1bp, with long positive. Durations above 50 or below 0, and negative DV01s, are treated as bad data.
- **Metrics:** zero rates, financing, borrow and cash interest. r_T = PnL_T / $10MM. ddof = 1, 252 days per year, every day included. Warm-up days are reported separately.
