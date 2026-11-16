# Deviations from the original scaffold

Branch `data-quality-and-research`, started from commit `dbc5d72` ("Add modular credit ETF backtester with pytest suite").
To see everything that changed: `git diff dbc5d72 --stat`.

Rule for this branch: where the scaffold differs from the project statement or from the group plan, the branch
follows the statement and the plan. Every change to scaffold behaviour is listed here with its reason and its test.
The scaffold's 97 tests passed before any change; the suite now has 157 tests and all pass (coverage 99%).

## A. Changes to scaffold behaviour

| ID | Files | What changed | Why | Effect on the supplied data | Test |
|---|---|---|---|---|---|
| D1 | `src/config.py`, `src/data_loader.py`, `src/context.py`, `src/risk.py`, `src/execution.py`, `src/backtester.py`, `main.py` (+ new `src/data_quality.py`) | New `BacktestConfig.clean_universe_only` (default **True**). ETFs with a missing dividend record or an unrecorded split are left out of `ctx.universe()`, and any target in them is forced to zero and logged as a rejected order with reason `excluded: incomplete corporate-action records`. `MarketData` carries a `quality` report; `daily` gains an `n_blocked` column; `main.py` prints the counts and writes `data_quality.json`. | The engine books PnL from recorded events only. The supplied dividend file has no cash records for most ETFs in Dec 2024 - Jul 2025, and two reverse splits are not recorded, so held funds book false losses or a false doubling. Statement section 2 makes the group responsible for the treatment of corporate actions; the group decided to exclude affected ETFs for the whole sample and handle them later in a separate imputation track. | 167 of 213 ETFs are excluded; 46 remain (list in `docs/data_gap.md`). With the flag off (original behaviour): a $10MM USFR holding books +$10.0MM on 2022-03-24; a $10MM BIL holding books -$32,806 on each of 2025-02-03, 2025-04-01, 2025-06-02 and -$40,461 on 2024-12-19; `equal_weight_long` receives $281k of distributions in Jan-Jul 2025 against $1.31MM in Jan-Jul 2024. With the flag on, `equal_weight_long` receives $1.30MM in both periods. | `tests/test_data_quality.py` (all classes); `tests/test_backtester.py::TestCorporateActionsInPnL::test_unrecorded_split_is_blocked_by_default` |
| D2 | `src/data_loader.py` (total-return formula) | `total_return = (split_ratio * close + cash) / prev_close - 1`. It was `(split_ratio * (close + cash)) / prev_close - 1`. | When a split and a cash distribution fall on the same date, the old formula counted the cash once per post-split share while the ledger (`Portfolio.apply_corporate_actions`) pays it once per pre-split share. The panel now matches the ledger. | None: no ETF has a split and a cash distribution on the same date. | `tests/test_backtester.py::TestCorporateActionsInPnL::test_split_and_dividend_on_the_same_day` |
| D3 | `src/performance.py` (`sharpe_ratio`), `src/config.py` (comment) | `rf` is an annualized rate; `rf / periods_per_year` is subtracted from each return. It was subtracted from each daily return unconverted. | A non-zero `risk_free_rate` would have been applied as a daily rate. | None at the default `risk_free_rate = 0`, which the statement fixes. | `tests/test_performance.py::TestRequiredMetrics::test_risk_free_rate_is_annualized` |
| D4 | `src/signal_analysis.py` (`ic_summary`, new `newey_west_tstat`) | For horizons above 1 day, `t_stat` uses a Newey-West variance with `horizon - 1` lags. The old value is kept as `t_stat_naive`. Horizon 1 is unchanged. `signal_ic.csv` gains the `t_stat_naive` column. | Daily ICs at a 5- or 20-day horizon share most of their forward window with their neighbours, so treating them as independent overstates the t-stat. | On a simulated 5-day overlap the t-stat goes from 4.55 to 2.48. | `tests/test_performance.py::TestSignalAnalysis::test_overlapping_horizon_t_stat_is_adjusted`, `::test_newey_west_with_no_lags_is_the_plain_t_stat` |
| D5 | `src/performance.py` (new `bill_rate_from_market`, `financing_charge`, `PerformanceAnalyzer.financing_adjusted`; `summary` takes an optional `bill_rate`), `src/backtester.py` (`BacktestResult.metrics` takes `bill_rate`), `main.py` | New metrics when a bill rate is available: `sharpe_net_of_financing`, `annualized_financing_charge`, `annualized_pnl_net_of_financing`, `avg_net_exposure`, `sharpe_audit_required`. The console prints `sharpe` as `sharpe_project_formula` and `gross_sharpe` as `gross_sharpe_project`, and prints an audit note when the net-of-financing Sharpe is above 1. Keys in `metrics.json` keep their names; the new keys are added. | The project formula sets r_f = 0 and financing = 0, so the full yield of a net-long book counts as excess return, and a T-bill ETF, which earns only the risk-free rate, scores far above 1. Group rule: the working metric charges the T-bill rate (SHV `yield` field, lagged one day) on the net dollar exposure; the project-formula value is shown only next to it; any Sharpe above 1 is audited before it is reported. | The required metrics are unchanged. For a dollar-neutral book the two Sharpe figures are equal. | `tests/test_performance.py::TestNetOfFinancing` |

## B. Existing tests that were changed

| Test | Change | Reason |
|---|---|---|
| `tests/conftest.py::vectorized_price_plus_dividend_pnl` | Uses the D2 formula. | It is the independent check of the ledger and has to use the ledger's convention. |
| `tests/test_backtester.py::TestCorporateActionsInPnL::test_split_without_adjustment_would_show_loss` | Runs with `clean_universe_only=False`. | The test builds a price that halves with no split record to show the raw loss. Under D1 that fund is detected and never held, which the new test `test_unrecorded_split_is_blocked_by_default` asserts. |

No other existing test was edited. New tests were added to two existing files: `tests/test_backtester.py`
(`test_unrecorded_split_is_blocked_by_default`, `test_split_and_dividend_on_the_same_day`) and
`tests/test_performance.py` (`test_risk_free_rate_is_annualized`, class `TestNetOfFinancing`,
`test_newey_west_with_no_lags_is_the_plain_t_stat`, `test_overlapping_horizon_t_stat_is_adjusted`).

## C. Additions (new files, no scaffold behaviour replaced)

| File | Contents |
|---|---|
| `src/data_quality.py` | Missing-dividend and unrecorded-split detection, the exclusion list, a distributions-vs-yield reconciliation, and cleaning helpers for the analytics fields (`clean_analytics`, `despike`). |
| `tests/test_data_quality.py` | 17 tests for the above and for the engine wiring. |
| `docs/data_gap.md` | Write-up of the corporate-action gaps for the team and the instructor. |
| `src/research/universe.py` | Bucket labels by ticker (assigned from fund names; the symbology has no category field), hedge anchors, reference ETF and recovery assumption per bucket. |
| `src/research/panels.py` | Ticker-labelled panels for the clean ETFs (returns close-to-close, close-to-3PM, 3PM-to-close; cleaned analytics) and the duration-matched Treasury hedge. |
| `src/research/stats.py` | Trailing z-score, half-life, forward sums, rank IC with optional within-bucket demeaning, IC tables and a time-series predictive regression with overlap-aware t-stats, and the net-of-financing Sharpe. Sits next to `SignalAnalyzer`; it does not replace it. |
| `src/research/hazard.py` | Credit triangle and a Nelson-Siegel curve fitted across ETFs (model edges, curve slope). |
| `src/research/merton.py` | Merton debt value, spread and debt volatility; solvers for leverage and asset vol; option-implied vol of the debt claim; rates-vs-credit vol split. |
| `src/research/signals.py` | Signal definitions indexed at the decision date: NAV discount (T-1 close and 3PM), 3PM factor residual, hazard-edge deviation, roll-down. |
| `tests/test_research.py` | 34 tests, including one per signal that corrupts everything a day-T decision cannot see and checks the signal through T is unchanged. |
| `notebooks/credit_etf_research.ipynb` | Re-measurement of every signal family on the clean ETFs. Committed without outputs (the repo is public and the outputs include per-ETF statistics of the supplied data fields); run it to regenerate the tables. |
| `requirements-notebook.txt` | `ipykernel`, `nbconvert`, `nbformat`, needed only to run the notebook. |
| `docs/results_audit.md` | Signal measurements, paper books, engine runs and the audit of each, with the assumptions behind every table. In-sample; not a final strategy result. |
| `scripts/audit/common.py`, `paper_books.py`, `hy_edge_audit.py`, `engine_runs.py` | Scripts that regenerate the tables in `docs/results_audit.md`. They use the engine through a lookup strategy defined in `common.py`; nothing is added to the strategy registry. Not covered by the test suite (outside `src/`). |
| `DEVIATIONS.md` | This file. |

`README.md` was edited to point here, to add `data_quality.json` to the outputs, and to add the D1, D4 and D5 assumptions.

## D. Checked against the project statement and left unchanged

- Calendar: dates in both the EOD and 3PM files; the six EOD-only dates are dropped (statement section 2).
- Universe: symbology ISINs only; other instruments are dropped from every file.
- Decision-time information: EOD through T-1, the day-T 3PM snapshot, corporate actions by `declared_date` (section 3). Day-T close and volume reach only the execution and portfolio code.
- Execution: fill at the day-T close; cap `floor(3% x day-T volume)`; $0.0035 per share traded on buys and sells; unfilled remainder cancelled.
- Capital: $10MM fixed; returns are PnL / $10MM; gross = sum of |shares x close|; sized to $57MM at 3PM against the $60MM cap.
- PnL: equity change, including distributions on `ex_date_adj` (shorts pay) and costs; splits rescale shares.
- Metrics (section 5): annualized PnL = 252 x mean; Sharpe with ddof = 1 and NaN when undefined; dollar and percent drawdown with V_0 in the running maximum; all days included, warm-up days counted separately.
- DV01: sum of shares x `dv01` / 100, in $ per -1bp, longs positive.

One correction to an earlier note: the README does include warm-up days in the metrics, as the statement requires.

## E. Judgment calls found in the audit and NOT changed (for the group to decide)

| # | Where | Observation |
|---|---|---|
| J1 | `src/risk.py` (untradable symbols) | A symbol with no day-T 3PM price keeps its position: it cannot be bought or sold that day, and it counts as zero in the 3PM gross calculation. PHB has no 3PM data after 2026-02-20 and GBF none after 2026-08-31, so a position in either could not be closed. Both are excluded under D1. |
| J2 | `src/backtester.py` (`gmv_breach`) | A gross value above $60MM after the close is flagged, not prevented. The $3MM buffer covers a 5.3% aggregate move between 3PM and the close. Listed as open in the scaffold's `PLAN.md`. |
| J3 | `src/portfolio.py` (`dv01`) | A position whose `dv01` field has never been populated contributes zero to the reported DV01. All 46 clean ETFs have the field from the start of the sample. |
| J4 | `src/data_loader.py` (`drop_duplicates`) | Four distribution rows are dropped as duplicates (GBIL 2022-05-03, JMUB and JMST 2021-12-30, JPIB 2022-02-01). They match on symbol, ex-date, type and amount but differ in another column. All four funds are excluded under D1. |
| J5 | `src/data_loader.py` (3PM price) | The decision price is the 3PM `last` when present, even if the bid/ask mid is more recent. |
| J6 | `src/signal_analysis.py` | IC uses raw total returns across all symbols, so for bond ETFs it is driven largely by duration. A rates-hedged, bucket-demeaned version is planned in `src/research/` as an addition. |
| J7 | `README.md` / `PLAN.md` | They name pandas 2.2. The suite passes on pandas 3.1.1, numpy 2.5.5, Python 3.12.16 (conda env `finm` created from `requirements.txt`). |

## F. Points for the instructor

1. The dividend file's missing records (Dec 2024 - Jul 2025) and the two unrecorded reverse splits; see `docs/data_gap.md`.
2. The exclusion in D1 is derived from the full sample. The statement says universe filters must use only data available when applied. This filter is on the integrity of the supplied corporate-action records, not on performance.
3. With zero financing and r_f = 0, a levered T-bill ETF position scores a very high project-formula Sharpe while earning only the risk-free rate.
