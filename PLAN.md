# Credit ETF Backtester — Development Plan

FINM 35700 Group Project (due Dec 8, 2026, 6PM CT). Goal: a modular, object-oriented backtester where a new
strategy is a single `Strategy` subclass, with pytest coverage proving the trading, capital and performance
rules are implemented correctly (the "Implementation correctness" item is 40/100 points).

Style is borrowed from FINM32500-Group3-Assignment-4: `src/` package + `tests/` + `main.py`, `requirements.txt`,
an `Order` with an `OrderState` enum, a risk-check gate, one pytest class per component, and CI with coverage ≥ 90%.
Dropped as overkill: FIX parser and generator, the singleton logger, and per-message prints. We use `logging` instead.

---

## 1. What the data says (verified 2026-11-16)

| Fact | Implication for the engine |
|---|---|
| Universe = 213 ISINs in `etf_symbology`; EOD has 224 symbols, 3PM has 226 | Filter every file to the symbology ISINs |
| EOD has 6 dates the 3PM file lacks (Columbus Day and Veterans Day 2020/21/25/26, which are bond holidays) | Trading calendar = intersection of the EOD and 3PM dates; drop those 6 "extra holidays" |
| 2020-01-02 → 2026-10-23, ~1,699 trading days | Full sample = in-sample period |
| 9 ETFs list late (IBTG/H/I/J, SGOV, EUSB, IBDV, IBTK, LQDB); none delist | Universe filters must be point-in-time (e.g. ≥ N days of history as of T-1) |
| Prices are **raw / unadjusted** (PREF went from 98.36 to 19.70 on its 5:1 split) | PnL must apply splits to positions; signals should use an adjusted total-return series |
| In-universe corp actions: Income 15,278, Omitted 165, LT/ST Cap Gain 208, Stock Split 9, Special Cash 2, Discontinued 3 | Cash types are Income, Cap Gains and Special Cash. Splits change the share count. Omitted and Discontinued are ignored |
| Split `dividend_amount` = share ratio (HYLB 1.25: 50 → 40) | q ← q·ratio at the T-1 close; never booked as cash |
| `declared_date ≤ ex_date_adj` always (41 rows declared on the ex date) | Strategy sees events with `declared_date ≤ T`; PnL applies all events on `ex_date_adj` |
| 3PM `last` missing on 0.9% of rows; median \|last/close − 1\| ≈ 4 bp | Fall back to the 3PM bid/ask mid |
| `dv01 ≈ close × duration / 100`; outliers (duration up to 13,703; negative dv01) | Clean or clip the risk fields; per-share $/bp = dv01/100 (verify) |
| Vol/option fields are 60–80% NaN | Optional signal inputs only |

---

## 2. Architecture

```
src/
  config.py         BacktestConfig (frozen dataclass): capital=10e6, gmv_cap=60e6, gmv_target_buffer,
                    adv_participation=0.03, cost_per_share=0.0035, periods_per_year=252, rf=0
  data_loader.py    DataLoader -> MarketData: universe filter, calendar alignment, wide date×symbol panels
                    (close, volume, bid/ask, 3PM last/mid, dv01, duration, spread, ...), corp-action table,
                    split-and-dividend-adjusted total-return index for signal research
  context.py        DecisionContext(T): the ONLY object a strategy sees.
                    eod (rows ≤ T-1) · snap_3pm (T) · known corp actions (declared ≤ T) · positions · capital
  strategy.py       Strategy (ABC): generate_targets(ctx) -> pd.Series (target shares)
                    + optional signal(ctx) -> pd.Series (scores for IC / RankIC)
                    Baselines: NoTrade, BuyAndHold, EqualWeightLong (equal $ notional of the GMV target)
  order.py          Order + OrderState(NEW, ACKED, PARTIALLY_FILLED, FILLED, REJECTED, CANCELED); Fill dataclass
  risk.py           RiskManager: scale targets so the 3PM-priced GMV ≤ cap - buffer; reject NaN/untradeable symbols
  execution.py      ExecutionHandler (MOC at close_T):
                    fill = sign × min(|order|, floor(0.03 × volume_T)); NaN or 0 volume (or NaN close) → no fill;
                    cost = 0.0035 × |filled shares|; unfilled remainder is cancelled (not carried over)
  portfolio.py      Portfolio/Ledger: (1) apply corp actions on ex_date_adj to T-1 positions (splits rescale q,
                    cash distributions credit longs and debit shorts), (2) trade at close_T, (3) mark to market:
                    PnL_T = Σ q_{T-1}(P_T − P_{T-1}) + dividends − costs [+ split-adjusted]; record GMV, DV01, cost, turnover
  backtester.py     Backtester(data, strategy, config).run() -> BacktestResult (daily frame + fills + orders)
  performance.py    PerformanceAnalyzer: the required metrics (see §4)
  signal_analysis.py SignalAnalyzer: IC, RankIC, ICIR, hit rate, decay (diagnostics; see §4)
main.py             run a named strategy, print metrics, save outputs
tests/              see §5
```

**Daily loop for date T:**
`ctx = DecisionContext(T)` → `targets = strategy.generate_targets(ctx)` → `risk.check(targets)` →
orders = targets − current q → `portfolio.apply_corporate_actions(T)` → `execution.fill(orders, close_T, volume_T)`
→ `portfolio.update(fills, close_T)` → record.

**Look-ahead protection is structural.** `DecisionContext` slices the panels to `< T` and exposes only the 3PM row
for T. The close and volume for T are visible only to `ExecutionHandler`/`Portfolio`.

**Adding a strategy** = one file in `strategies/` subclassing `Strategy`, then `python main.py --strategy MyStrat`.

---

## 3. Capital and execution assumptions

- $10MM fixed capital; returns r_T = PnL_T / 10MM (no compounding); zero rates, financing, borrow and cash interest.
- GMV_T = Σ|q·close_T| ≤ $60MM. Sizing uses 3PM prices, so the close can drift past the cap. Use a buffer
  (e.g. size to $57–58MM), and test that GMV stays ≤ $60MM after the fill on every day.
- 3% of the day-T volume cap, $0.0035/share both sides, MOC at the official close.
- Whole shares (rounded toward zero).

---

## 4. Performance metrics

**Required (rubric):** daily net $PnL, GMV, aggregate DV01 ($ per −1bp, longs positive), annualized net PnL = 252·mean,
Sharpe = √252·mean/sd (ddof=1, NaN if sd=0 or n<2), max $ drawdown max(M−V), max % drawdown max(1−V/M),
with V_0 = 10MM included in the running max. Every trading day counts, including inactive days; the warm-up is reported separately.

**Proposed diagnostics (for discussion):**
| Metric | Definition | Use |
|---|---|---|
| IC | Pearson corr(signal_T, fwd total return close_T→close_{T+1}) across ETFs, per day | Signal quality, value-weighted |
| Rank IC | Spearman version | Ranking/cross-sectional strategies |
| ICIR | mean(IC)/sd(IC) (×√252 annualized) | Signal consistency |
| IR | mean(r − r_bench)/sd(r − r_bench)·√252 | Needs a benchmark (AGG? equal-weight? none?) |
| Also | turnover, cost drag, avg GMV utilisation, hit rate, IC decay at 1/5/20 days, DV01-neutral check | Presentation |

Forward returns for IC use the split- and dividend-adjusted series. Signals are aligned to the information
available at the 3:45PM decision on T.

---

## 5. Test plan (pytest, synthetic fixtures in `conftest.py`, 2–3 symbols × ~10 days)

| Test class | Asserts |
|---|---|
| TestDataLoader | 6 holidays removed, EOD/3PM aligned, only universe ISINs, 3PM mid fallback |
| TestNoLookahead | Poisoning close_T/volume_T (and every row > T-1 of the EOD) leaves the decisions at T unchanged |
| TestTrivialStrategy | NoTrade → PnL ≡ 0, Sharpe NaN, DD = 0, every date present |
| TestExecution | fill = floor(3% vol); zero/NaN volume → no fill; cost symmetric for buys and sells; partial → PARTIALLY_FILLED |
| TestRisk | GMV ≤ $60MM after the fill every day; targets scaled proportionally |
| TestCorporateActions | split leaves the economic value unchanged; dividend only on ex_date_adj; long receives and short pays; Omitted ignored |
| TestPortfolio | Σ PnL = final equity − 10MM; mark-to-market by hand on a toy series |
| TestPerformance | Sharpe and $/% drawdown match hand calculations; V_0 in the running max; ddof=1 |
| TestDV01 | sign and units on a known position |
| TestSignalAnalysis | IC = 1 for a perfect-foresight signal; RankIC matches scipy spearman |
| test_smoke (slow mark) | Full real-data run of EqualWeightLong finishes and satisfies every invariant |

CI: `.github/workflows/ci.yml`, `coverage run -m pytest -q`, `--fail-under=90`.

---

## 6. Milestones

1. **Data layer + tests**: DataLoader, MarketData, calendar, corp-action table, adjusted returns.
2. **Engine core + tests**: Context, Strategy ABC, Order, Risk, Execution, Portfolio, Backtester; baseline strategies.
3. **Analytics + tests**: PerformanceAnalyzer, SignalAnalyzer, plots (PnL, drawdown, GMV, DV01).
4. **Smoke run on real data**; README with run instructions (`envs/finm` Python 3.12, pandas 2.2).
5. **Strategy research** (group): candidates include spread/carry, momentum, 3PM-vs-NAV dislocation,
   fund-flow reversal and duration-hedged credit spread. Each is a `Strategy` subclass evaluated by IC first, then the full backtest.
6. Presentation outputs.

## 7. Decisions

**Decided (2026-11-16):**
- Diagnostics: IC, Rank IC, ICIR, IR, turnover/cost drag, IC decay (1/5/20d). All of them.
- IR benchmark: AGG buy & hold scaled to $10MM.
- Unfilled remainder: cancelled; the strategy re-targets the next day from its actual positions.

**Defaulted during the build (all are config knobs in `BacktestConfig` and can be changed):**
- GMV: size to $57MM at 3PM, plus a worst-case check (`worst_case_gmv_check=True`); a post-trade breach is flagged in `daily.gmv_breach`.
- Whole shares (`whole_shares=True`); a split can leave fractional holdings, which are kept.
- Strategy output: target shares (`Strategy`) or weights of the GMV target (`WeightStrategy`). Both are supported.
- Universe: `ctx.universe(min_history=20)` helper; each strategy chooses its own filters.

**Status (2026-11-16):** milestones 1–4 are done: 96 tests pass, including the real-data smoke run. Ledger PnL matches an independent
vectorized total-return calculation to within $1e-7 over the full sample.

**Original open list (kept for reference):**
3. GMV buffer size and how to handle a post-fill breach (none is possible to prevent at the close; size conservatively).
4. Universe filter defaults (min history, min ADV, spread cap).
5. Whole vs fractional shares.
6. Strategy output convention: target shares vs target $ weights.
