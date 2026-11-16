# Signal measurements and audit (in-sample, clean ETFs)

Written 2026-11-16 on branch `data-quality-and-research`. Sample: 2020-01-02 to 2026-10-23, the full supplied
sample, so every number here is in-sample. Nothing in this document is a final strategy result. It records what
was measured, under which assumptions, and what the checks on those measurements found.

Every table can be regenerated with the scripts in `scripts/audit/` (section 12).

**Sharpe convention.** Unless a column says otherwise, Sharpe is net of financing: the T-bill rate is charged on
the book's net dollar exposure. Values above 1 are marked † and are not treated as results; the check made on each
is cited next to it.

## 1. What was measured

| Family | Lecture | Signal known at the day-T decision |
|---|---|---|
| NAV basis | 2 | Discount of price to NAV, z-score against the fund's trailing 60 days: at the T-1 close, and from the day-T 3PM mid |
| 3PM factor residual | 5 (risk model) | Minus the T-1 close to 3PM move left after the rates part and beta x the reference ETF (HYG or LQD) |
| Hazard-curve edge | 4-5 | Market g-spread minus a Nelson-Siegel curve fitted across the funds of a tier; raw, and as a z-score against its own trailing mean |
| Roll-down | 1 | (carry + duration x curve slope) per year of duration, for IG spreads and for Treasury yields |
| Merton tier model | 6 | Each fund's implied log leverage (using HYG's solved asset vol) against its own trailing 60 days; benchmark: spread gap to HYG |
| Pairs against HYG | 4 (seniority) | Trailing-120d z-score of log(h_leg / h_HYG), h = g-spread / (1 - R) |
| Merton state variables | 6 | Trailing-120d z-scores of HYG's solved leverage and asset vol; vol term-structure gap |

## 2. Assumptions

### 2.1 Data and universe
- The 46 ETFs with complete corporate-action records (`docs/data_gap.md`). The exclusion list is built from the
  full sample.
- Instruments outside the symbology file are not used, as positions or as signal inputs.
- Analytics fields are range-filtered; spread fields also pass a trailing spike filter (`src/data_quality.py`).
  The filter leaves spreads missing on some days, mostly 12-23 March 2020.
- Buckets are assigned from fund names (`src/research/universe.py`). "Liquid" means a 3% cap of at least $1MM a
  day at the full-sample median (18 credit ETFs).

### 2.2 Timing
- A signal dated T uses EOD data through T-1, plus the day-T 3PM snapshot where stated.
- A book decided at T is entered at the close of T and earns the close-to-close return of T+1.
- `tests/test_research.py` corrupts everything a day-T decision cannot see and checks each signal is unchanged.

### 2.3 Rates hedge
- Hedged return = total return minus a duration-matched Treasury return interpolated from SHY, IEI, IEF, TLH, TLT.
- The hedge is matched in duration, not in dollars. Hedge dollars per $1 of fund are between about 1.0 and 1.2
  for funds inside the anchor range.
- In the paper books the hedge ratio for the return of T+1 uses the duration of T, one day fresher than the
  decision allows. Lagging it one more day changes the HY edge book's Sharpe from 0.05 to 0.07 (section 7). The
  engine runs use durations through T-1.
- BKLN and SRLN hold floating-rate loans but carry a duration near 3.6 in the data. Loan results are shown with
  the data duration and with a duration of 0.25.

### 2.4 Paper books
- Cross-sectional book: weights = demeaned cross-sectional rank of the signal, scaled to sum |w| = 1. NAV and 3PM
  residual books are demeaned within bucket. Roll-down weights are in duration units and converted to dollars by
  dividing by duration.
- Pair book: position = clip(z, -3, 3) / 3 in the leg and the opposite position in HYG.
- Holding period h: the average of the last h days' weight vectors, rescaled to $1 of gross.
- Missing signal: the signal is carried forward for up to 10 days and every fund is re-ranked, so each day's book
  nets to zero. A day with fewer than 5 funds keeps the whole previous book.
- No volume cap.

### 2.5 Costs and financing
- $0.0035 per share on the fund legs and on the Treasury hedge legs, at closing prices. For the 3PM residual books
  only the fund legs are costed, because their target is a factor residual.
- Financing: the T-bill rate (SHV `yield` field, lagged one day) on the net dollars of all legs. A net-short book
  is credited. The project statement sets financing and cash interest to zero; its formula is reported as
  "project formula" in the engine tables.
- Long-only reference for the convention (buy-and-hold, Sharpe net of financing): SHV -0.78, SHY -0.63, IEF -0.41,
  TLT -0.38, VCSH -0.10, LQD -0.19, FLOT 0.13, HYG 0.09, BKLN 0.22, SJNK 0.27.

### 2.6 Engine runs
- Weights for decision date T are precomputed from data through T-1 and handed to the scaffold engine as a lookup
  (`scripts/audit/common.py::Lookup`). The engine's structural information guard is therefore not exercised; the
  lags are explicit in the scripts.
- The engine supplies: $10MM capital, fills at the day-T close, the 3% cap on day-T volume, $0.0035 per share on
  every leg including hedges, corporate actions, the $57MM gross target and the $60MM cap.
- Books are scaled so their largest target gross is 0.9 to 0.95 of the $57MM target. The pair books use much less
  than that on average because positions scale with |z|.

### 2.7 Choices made on the full sample
- Curve location parameter: 3 for IG (three Nelson-Siegel terms), 1 for HY (two terms), chosen by median fit error.
- Windows: 60 days (NAV z, tier model), 120 days (edge deviation, pair z, state variables), 120-day beta.
- Recovery: 0.80 loans, 0.40 high-yield bonds.
- The clean-universe list and the liquid list.

### 2.8 Not included
- Out-of-sample tests. The statement defines none.
- The 3% cap in the paper books.
- The 167 ETFs with incomplete corporate-action records (planned as a separate imputed track).

## 3. Reading the statistics

- **IC t-stat.** Each day the funds are ranked by signal and by forward hedged return and correlated. The t-stat
  is the mean daily correlation over its standard error, with a Newey-West variance (horizon - 1 lags) at 5 and 20
  days. Every day counts equally and return sizes are ignored.
- **PnL t-stat.** For a daily PnL series t = Sharpe x sqrt(years). The sample is about 6.6 years, so t is about
  2.6 x Sharpe: a Sharpe of 0.8 corresponds to t = 2. The standard error of a Sharpe on this sample is about 0.39.
- **The two can differ.** The NAV discount at the T-1 close has a 1-day IC t-stat of 3.0 and a 1-day paper book
  with t = 1.3 before costs. PnL weights days by the size of the moves.
- **Number of tests.** About 25 signal variants were measured at 3 horizons. With about 75 tests, roughly 4 would
  show |t| > 2 if no signal had any effect.

## 4. Information coefficients

Mean daily rank IC with the forward rates-hedged return, t-stat in brackets. Before costs.

| Signal | ETFs | 1 day | 5 days | 20 days |
|---|---|---|---|---|
| NAV discount at T-1 close | 25 | 0.022 (3.0) | 0.029 (2.7) | 0.040 (2.6) |
| NAV discount at T-1 close, liquid | 18 | 0.020 (2.3) | 0.026 (2.2) | 0.031 (1.8) |
| NAV discount at 3PM | 25 | 0.023 (2.4) | 0.036 (2.8) | 0.021 (1.2) |
| NAV discount at 3PM, liquid | 18 | 0.010 (1.0) | 0.024 (1.7) | 0.001 (0.1) |
| 3PM residual reversal | 23 | 0.021 (2.7) | 0.007 (0.8) | 0.006 (0.7) |
| 3PM residual reversal, liquid | 16 | 0.013 (1.5) | -0.002 (-0.2) | -0.004 (-0.4) |
| Hazard edge deviation z, IG | 13 | 0.000 (0.0) | 0.022 (1.9) | 0.034 (1.6) |
| Hazard edge deviation z, HY | 6 | 0.009 (0.9) | -0.008 (-0.6) | -0.030 (-0.9) |
| Hazard raw edge, IG | 13 | 0.012 (1.7) | 0.035 (3.1) | 0.055 (2.5) |
| Hazard raw edge, HY | 6 | 0.014 (1.5) | 0.050 (3.6) | 0.120 (3.4) |
| IG spread carry + roll-down | 13 | 0.036 (2.8) | 0.110 (4.2) | 0.179 (3.2) |
| IG spread roll-down only | 13 | 0.029 (2.3) | 0.095 (3.7) | 0.153 (2.8) |
| Rates carry + roll-down | 8 | -0.003 (-0.2) | -0.017 (-0.7) | -0.038 (-0.8) |
| Merton leverage edge vs HYG | 7 | 0.014 (1.1) | -0.007 (-0.4) | 0.006 (0.2) |
| Naive spread-gap edge vs HYG | 7 | 0.022 (1.9) | 0.015 (0.8) | 0.009 (0.3) |

- Targets: the hedged return; for the 3PM residual, the same residual on the following close-to-close returns; for
  roll-down, the return per year of duration; for the last two rows, the return relative to HYG.
- The IC of the 3PM residual with the 3PM-to-close residual of the same day is 0.151. That part is before the fill.
- Carrying a missing signal forward adds 10 to 28 days to the spread-based series and leaves the values almost
  unchanged (IG carry + roll-down at 5 days: 0.110 (4.2) becomes 0.103 (3.9)).
- For the hazard raw edge, removing the accrual of the edge itself from the target gives 0.023 (2.0) at 5 days for
  IG and -0.008 for HY (research notebook, section 5).

Time-series relations from the research notebook (correlation of the signal with the forward return, z-scores
clipped at +/-3, t-stat in brackets):

| Signal | 1 day | 5 days | Note |
|---|---|---|---|
| HYG solved leverage z | -0.025 (-0.9) | | Unclipped: -0.109 |
| HYG solved asset vol z | 0.006 (0.2) | | Unclipped: 0.116; 97% of the covariance comes from 10 days in 2020 and 2022; 0.006 without them |
| HYG vol term-structure gap, 12M | -0.052 (-1.6) | -0.056 (-1.1) | |
| BKLN vs HYG hazard gap z | 0.053 (1.6) | 0.119 (2.6) | |
| SRLN vs HYG hazard gap z | 0.059 (1.7) | 0.078 (1.7) | |
| SPHY vs HYG hazard gap z | 0.071 (2.8) | 0.160 (4.0) | |
| USHY vs HYG hazard gap z | 0.049 (2.1) | 0.085 (2.1) | |
| JNK vs HYG hazard gap z | 0.038 (1.4) | 0.041 (0.9) | |
| SHYG vs HYG hazard gap z | -0.026 | -0.073 | |
| SJNK vs HYG hazard gap z | -0.023 | -0.034 | |

The Merton calibration (leverage and asset vol from the spread and the credit-only option-implied vol) solved on
94.7% of days and gave a single solution on each of 41 dates checked from six starting points.

## 5. Cross-sectional paper books

Sharpe net of financing, per $1 of fund gross, before the volume cap. "After costs" includes commission on fund
and hedge legs. Source: `scripts/audit/paper_books.py`.

| Signal | ETFs | Before costs, 1d | After costs, 1d | 5d | 20d | 60d | t, 20d | 20d excl. 2020 | Turnover per day, 1d |
|---|---|---|---|---|---|---|---|---|---|
| NAV discount at T-1 close | 25 | 0.51 | -1.58 | 0.01 | -0.23 | 0.14 | -0.6 | -0.56 | 1.01 |
| NAV discount at T-1 close, liquid | 18 | 0.33 | -1.32 | 0.07 | -0.20 | 0.13 | -0.5 | -0.47 | 1.01 |
| NAV discount at 3PM | 25 | 0.36 | -1.11 | 0.00 | -0.19 | -0.07 | -0.5 | -0.70 | 0.99 |
| NAV discount at 3PM, liquid | 18 | 0.14 | -1.20 | 0.00 | -0.22 | -0.08 | -0.6 | -0.74 | 1.00 |
| 3PM residual reversal | 23 | 1.07 † | -5.35 | -2.30 | -1.18 | -0.67 | -3.0 | -1.25 | 1.29 |
| 3PM residual reversal, liquid | 16 | 0.70 | -4.59 | -2.19 | -1.27 | -0.85 | -3.2 | -1.22 | 1.28 |
| Hazard edge deviation z, IG | 13 | 0.41 | -1.18 | -0.17 | 0.23 | 0.10 | 0.6 | 0.32 | 0.60 |
| Hazard edge deviation z, HY | 6 | 0.66 | -1.37 | -0.33 | -0.22 | -0.26 | -0.6 | -0.36 | 0.44 |
| Hazard raw edge, IG | 13 | -0.10 | -0.70 | 0.12 | 0.38 | 0.24 | 1.0 | 0.34 | 0.24 |
| Hazard raw edge, HY | 6 | 0.05 | -0.19 | 0.04 | 0.19 | 0.30 | 0.5 | 0.23 | 0.09 |
| IG spread carry + roll-down | 13 | 0.22 | 0.17 | 0.22 | 0.29 | 0.30 | 0.7 | 0.69 | 0.05 |
| IG spread roll-down only | 13 | 0.18 | 0.11 | 0.23 | 0.22 | 0.23 | 0.6 | 0.62 | 0.05 |
| Rates carry + roll-down | 8 | -0.49 | -0.90 | -0.57 | -0.49 | -0.45 | -1.3 | -0.54 | 0.26 |
| Merton leverage edge vs HYG | 7 | -0.16 | -0.91 | -0.41 | -0.18 | -0.16 | -0.5 | -0.28 | 0.53 |
| Naive spread-gap edge vs HYG | 7 | 0.85 | 0.06 | 0.04 | -0.06 | 0.12 | -0.2 | -0.21 | 0.51 |

† Before costs only; after commission on the fund legs it is -5.35 (section 9).

Sharpe by year, 20-day hold, after costs:

| Signal | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|
| NAV discount at T-1 close | 0.00 | 0.51 | -1.73 | -0.36 | 1.09 † | -0.33 | -1.65 |
| NAV discount at 3PM | 0.16 | -0.31 | -1.38 | -0.08 | 0.06 | -0.10 | -2.33 |
| 3PM residual reversal | -1.29 | -2.02 | -0.54 | -1.97 | -1.26 | -0.68 | -2.12 |
| Hazard edge deviation z, IG | -0.28 | 0.41 | 0.38 | 0.39 | 0.09 | 0.87 | -0.35 |
| Hazard edge deviation z, HY | 0.52 | -1.49 | 0.03 | 0.23 | -0.08 | -0.25 | -1.68 |
| Hazard raw edge, IG | 0.65 | -0.22 | 0.44 | 1.02 † | 0.32 | -0.14 | 1.23 † |
| Hazard raw edge, HY | 0.23 | 1.08 † | -0.45 | 1.38 † | 0.47 | -0.31 | -0.93 |
| IG spread carry + roll-down | 0.24 | -0.17 | 0.55 | 0.43 | 1.74 † | 1.34 † | 1.34 † |
| IG spread roll-down only | 0.08 | -0.19 | 0.50 | 0.36 | 1.58 † | 1.26 † | 1.20 † |
| Rates carry + roll-down | -0.17 | -1.74 | -1.52 | 0.29 | 0.44 | -0.97 | -0.26 |
| Merton leverage edge vs HYG | 0.05 | 0.22 | -0.10 | -0.93 | -0.03 | -0.26 | -0.42 |
| Naive spread-gap edge vs HYG | 0.28 | -0.71 | -0.13 | 0.06 | -0.56 | -0.30 | 0.18 |

† Single-year values. The IG carry + roll-down years are examined in section 8.3: they come with annual PnL of
$49k to $119k on $10MM and low volatility, and the same book lost $1.6MM over five days in March 2020.

## 6. Pairs against HYG

Position = clip(z, -3, 3) / 3 in the leg, the opposite in HYG, both rates-hedged. BKLN and SRLN are loan funds, so
those two pairs compare seniority ranks. SPHY, USHY, JNK, SHYG and SJNK are high-yield bond funds like HYG; those
pairs compare funds of the same rank. Source: `scripts/audit/paper_books.py`.

| Pair | Loan duration | Before costs, 1d | After costs, 1d | 5d | 20d | 60d | t, 1d | 1d excl. 2020 |
|---|---|---|---|---|---|---|---|---|
| BKLN vs HYG | data (about 3.6) | 0.64 | 0.42 | 0.34 | 0.30 | 0.58 | 1.1 | 0.61 |
| BKLN vs HYG | 0.25 | 0.52 | 0.29 | 0.20 | 0.16 | 0.25 | 0.7 | 0.46 |
| SRLN vs HYG | data (about 3.6) | 0.70 | 0.56 | 0.48 | 0.24 | 0.23 | 1.4 | 0.29 |
| SRLN vs HYG | 0.25 | 0.69 | 0.51 | 0.52 | 0.40 | 0.40 | 1.3 | 0.20 |
| SPHY vs HYG | | 1.03 † | 0.33 | 0.59 | 0.25 | -0.05 | 0.8 | -0.40 |
| USHY vs HYG | | 0.90 | 0.06 | 0.20 | 0.14 | -0.01 | 0.2 | -0.62 |
| JNK vs HYG | | 0.71 | -0.12 | 0.03 | 0.11 | -0.06 | -0.3 | -0.22 |
| SHYG vs HYG | | -0.15 | -0.50 | -0.41 | -0.52 | -0.35 | -1.3 | -0.97 |
| SJNK vs HYG | | -0.23 | -0.92 | -0.40 | -0.31 | -0.14 | -2.4 | -1.51 |

† Before costs. By year after costs it is 1.97 in 2020 and negative in 2021 and in 2023-2026 (table below). SPHY's
3% cap was $0.03MM a day in 2020 and $0.10MM in 2021 (median).

Sharpe by year, 1-day hold, after costs:

| Pair | Loan duration | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|
| BKLN vs HYG | data | 0.41 | 0.81 | 1.28 † | 0.30 | 1.07 † | 0.52 | -0.70 |
| BKLN vs HYG | 0.25 | 0.28 | 0.26 | 1.07 † | 1.01 † | 0.51 | 0.37 | -1.44 |
| SRLN vs HYG | data | 1.42 † | 1.40 † | 0.24 | 0.15 | 0.95 | 0.24 | -1.18 |
| SRLN vs HYG | 0.25 | 1.39 † | 0.96 | 0.56 | 0.82 | -0.74 | 0.44 | -2.16 |
| SPHY vs HYG | | 1.97 † | -0.20 | 0.49 | -0.26 | -1.92 | -3.11 | -1.80 |
| USHY vs HYG | | 2.22 † | -0.52 | 0.51 | -1.01 | -1.07 | -2.53 | -1.65 |
| JNK vs HYG | | 0.05 | 0.53 | 1.45 † | -0.61 | -1.52 | -2.01 | -0.47 |
| SHYG vs HYG | | 0.61 | -0.61 | -1.12 | -1.17 | -1.28 | -0.76 | -2.13 |
| SJNK vs HYG | | 0.53 | -1.20 | -1.56 | -1.21 | -2.74 | -1.69 | -2.56 |

† Single-year values; the full-sample figures are in the table above.

## 7. Audit of the HY hazard raw-edge book

Source: `scripts/audit/hy_edge_audit.py`. Six funds: HYG, JNK, USHY, SPHY, SHYG, SJNK.

**Missing signals.** 157 fund-days have no signal, 86 of them in March 2020. The result depends on what the book
does on those days:

| Treatment of a fund with no signal | Before costs, 1d | After costs, 1d | 5d | 20d | 60d | 120d |
|---|---|---|---|---|---|---|
| A. Signal carried forward, all six funds re-ranked (used in this document) | 0.05 | -0.19 | 0.04 | 0.19 | 0.30 | 0.29 |
| B. No position in that fund | 0.60 | 0.18 | 0.48 | 0.38 | 0.42 | 0.42 |
| C. That fund keeps its previous weight while the others are re-ranked | 1.02 † | 0.77 | 0.61 | 0.23 | 0.32 | 0.31 |

† Treatment C does not net to zero: it carried net high-yield exposure through March 2020. An earlier pass of this
work used C and reported 0.80 after fund-leg commission. That figure is withdrawn.

Under treatment A:

- The ten largest daily PnL moves are all between 9 March and 16 April 2020 (-97bp to +68bp of gross). March 2020
  PnL is -3.07% of gross.
- Sharpe after costs excluding 2020: -0.07 (1d), 0.22 (5d), 0.23 (20d), 0.25 (60d). Excluding only March and April
  2020: 0.37, 0.60, 0.61, 0.62.
- Sharpe by year after costs, 60-day hold: 0.55, 1.20 †, -0.54, 1.36 †, 0.47, -0.28, -0.95.

**What the book holds.**

| Fund | Mean weight | SD of weight | Share of days long | Mean edge, bp |
|---|---|---|---|---|
| SJNK | 0.236 | 0.077 | 99.0% | 35 |
| SPHY | 0.177 | 0.067 | 99.6% | 27 |
| USHY | 0.011 | 0.075 | 52.7% | 13 |
| JNK | 0.005 | 0.106 | 47.0% | 10 |
| HYG | -0.152 | 0.044 | 1.5% | -11 |
| SHYG | -0.277 | 0.011 | 0.0% | -46 |

- Holding the full-sample average weights every day gives a Sharpe of 0.45 before costs. The deviations from those
  weights give -0.32. (The average uses the full sample; this is a decomposition, not a tradable rule.)
- A tradable split gives the same picture: ranking on the trailing 250-day mean of the edge gives 0.43 after costs
  at a 1-day hold (0.27 excluding 2020); ranking on the edge minus that mean gives 0.10 before costs and -1.33 after.
- The book's spread pickup averages 0.26% of gross a year. Its realised return at a 1-day hold is 0.05% a year.

**Timing checks.**

- Hedge duration lagged one more day: Sharpe 0.07 against 0.05. Entry one day later: -0.11. No rates hedge: 0.04.
- Correlation of the day-t g-spread change with the hedged return of day t+k:

| Fund | k = -1 | k = 0 | k = +1 |
|---|---|---|---|
| HYG | 0.080 | -0.948 | 0.021 |
| JNK | 0.086 | -0.937 | 0.008 |
| SPHY | 0.108 | -0.873 | 0.048 |
| SHYG | 0.110 | -0.836 | 0.039 |
| USHY | 0.058 | -0.819 | 0.020 |
| SJNK | 0.134 | -0.817 | 0.009 |
| BKLN | 0.017 | -0.788 | 0.062 |
| SRLN | 0.051 | -0.790 | 0.038 |

The spread change lines up with the same day's return and has a correlation of 0.01 to 0.06 with the next day's.

## 8. Engine runs

Source: `scripts/audit/engine_runs.py`. $10MM capital, 3% cap, fills at the close, commission on every leg. No run
breached the $60MM gross cap.

### 8.1 Loans against HYG

BKLN and SRLN together, each with its own position; HYG takes the other side.

| Hold | Loan duration | Avg gross, $MM | Net PnL, $k a year | Costs, $k a year | Sharpe, project formula | Sharpe net of financing | t | Excl. 2020 | Max drawdown |
|---|---|---|---|---|---|---|---|---|---|
| 1 day | data | 18.4 | 362 | 133 | 0.49 | 0.49 | 1.3 | 0.55 | 5.5% |
| 20 days | data | 14.9 | 145 | 21 | 0.32 | 0.32 | 0.8 | 0.43 | 7.4% |
| 30 days | data | 14.0 | 146 | 15 | 0.39 | 0.38 | 1.0 | 0.48 | 4.8% |
| 60 days | data | 12.1 | 132 | 9 | 0.48 | 0.47 | 1.2 | 0.54 | 2.8% |
| 90 days | data | 10.7 | 104 | 6 | 0.44 | 0.42 | 1.1 | 0.49 | 2.7% |
| 120 days | data | 9.7 | 73 | 5 | 0.34 | 0.33 | 0.9 | 0.41 | 2.6% |
| 1 day | 0.25 | 14.9 | 187 | 102 | 0.39 | 0.38 | 1.0 | 0.40 | 4.0% |
| 20 days | 0.25 | 12.1 | 96 | 15 | 0.33 | 0.31 | 0.8 | 0.52 | 5.6% |
| 30 days | 0.25 | 11.4 | 104 | 11 | 0.44 | 0.41 | 1.1 | 0.60 | 3.7% |
| 60 days | 0.25 | 9.8 | 76 | 6 | 0.46 | 0.40 | 1.1 | 0.53 | 1.9% |
| 90 days | 0.25 | 8.6 | 52 | 4 | 0.36 | 0.31 | 0.8 | 0.43 | 1.9% |
| 120 days | 0.25 | 7.8 | 39 | 3 | 0.29 | 0.25 | 0.7 | 0.39 | 1.8% |

Sharpe by year, net of financing:

| Hold | Loan duration | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|
| 1 day | data | 0.64 | 1.34 † | 0.96 | 0.23 | 1.23 † | 0.39 | -0.94 |
| 20 days | data | 0.16 | 0.49 | 0.52 | 0.20 | 0.79 | 0.10 | 0.65 |
| 60 days | data | 0.28 | 0.21 | 0.25 | 1.34 † | -0.40 | 0.34 | 1.58 † |
| 1 day | 0.25 | 0.58 | 0.81 | 1.01 † | 0.88 | -0.20 | 0.41 | -1.88 |
| 20 days | 0.25 | 0.03 | 0.32 | 1.23 † | 0.36 | 0.54 | 0.66 | -0.44 |
| 60 days | 0.25 | 0.13 | -0.25 | 1.21 † | 1.04 † | -0.01 | 0.75 | 0.32 |

† Single-year values; full-sample figures are in the table above.

Single legs at a 20-day hold (Sharpe net of financing, data duration / duration 0.25): BKLN 0.31 / 0.17; SRLN 0.18 /
0.37. The paper books give 0.30 / 0.16 and 0.24 / 0.40.

With the 0.25 duration the engine reports an average DV01 of about +$300 per bp, because it values BKLN and SRLN
with the data's DV01 field.

### 8.2 HY hazard raw edge

| Hold | Avg gross, $MM | Net PnL, $k a year | Costs, $k a year | Sharpe, project formula | Sharpe net of financing | t | Excl. 2020 | Max drawdown | Worst day, $k |
|---|---|---|---|---|---|---|---|---|---|
| 1 day | 51.3 | -24 | 67 | -0.04 | -0.04 | -0.1 | -0.19 | 17.0% | -621 |
| 60 days | 52.2 | 127 | 14 | 0.21 | 0.22 | 0.6 | 0.20 | 14.1% | -575 |

### 8.3 IG spread carry + roll-down

| Hold | Net PnL, $k a year | Costs, $k a year | Sharpe, project formula | Sharpe net of financing | t | Excl. 2020 | Excl. Mar-Apr 2020 | Max drawdown | Worst day, $k |
|---|---|---|---|---|---|---|---|---|---|
| 1 day | 38 | 20 | 0.08 | 0.18 | 0.5 | 0.59 | 0.76 | 20.6% | -583 |
| 5 days | 50 | 11 | 0.11 | 0.22 | 0.6 | 0.67 | 0.84 | 19.1% | -552 |
| 20 days | 95 | 8 | 0.22 | 0.34 | 0.9 | 0.68 | 0.85 | 17.9% | -518 |
| 30 days | 90 | 8 | 0.21 | 0.33 | 0.9 | 0.68 | 0.83 | 17.4% | -508 |
| 60 days | 93 | 8 | 0.22 | 0.34 | 0.9 | 0.68 | 0.91 | 17.4% | -506 |
| 90 days | 92 | 8 | 0.22 | 0.34 | 0.9 | 0.68 | 0.91 | 17.4% | -505 |
| Fixed average weights | 76 | 4 | 0.20 | 0.34 | 0.9 | 0.73 | 0.86 | 14.8% | -433 |

- Average gross is about $52MM in every run. The book is net short about $1.8MM, so the net-of-financing Sharpe
  includes an interest credit that the project formula does not.
- Sharpe by year, 20-day hold, net of financing: 0.33, -0.11, 0.52, 0.44, 1.63 †, 1.33 †, 1.37 †.
- PnL by year, 90-day hold, $k: 324, -16, 54, 11, 81, 119, 49 (2026 through October).
- 2020 by month, $k: March -369, April +345, May +208. The five worst days are 12-19 March 2020, between -$192k and
  -$505k each. The maximum drawdown is $1.75MM.
- † The three years above 1 have annual PnL of $49k to $119k.
- The paper book gives 0.29 after costs at a 20-day hold.

What the book holds (90-day run, average position):

| Fund | Mean weight, 1-day book | Share of days long | Average position, $MM | 3% cap 2020, $MM a day | 3% cap 2026, $MM a day |
|---|---|---|---|---|---|
| SPSB | 0.322 | 100% | 8.40 | 1.11 | 2.54 |
| IGSB | 0.168 | 100% | 4.39 | 3.23 | 4.84 |
| SCHJ | 0.135 | 99.9% | 3.46 | 0.02 | 0.15 |
| VCSH | 0.122 | 100% | 3.21 | 6.42 | 11.26 |
| SPIB | 0.051 | 100% | 1.32 | 1.55 | 6.83 |
| IGIB | 0.013 | 83.7% | 0.34 | 2.13 | 4.30 |
| SCHI | -0.005 | 14.9% | -0.12 | 0.02 | 1.58 |
| VCIT | -0.011 | 2.0% | -0.28 | 8.11 | 24.38 |
| USIG | -0.022 | 0.9% | -0.57 | 0.93 | 3.00 |
| IGLB | -0.030 | 0.0% | -0.80 | 0.44 | 2.59 |
| LQD | -0.032 | 0.4% | -0.85 | 49.39 | 99.27 |
| SPLB | -0.037 | 0.0% | -0.96 | 0.48 | 2.14 |
| VCLT | -0.043 | 0.0% | -1.12 | 1.90 | 11.35 |

- Hedge legs, average $MM: SHY -19.3, IEI -3.3, IEF +1.5, TLH +2.8, TLT +0.1.
- Fills: 1.7% of target gross was unfilled on average in 2020 and none afterwards. SCHJ took from January to May
  2020 to reach its target ($0.04MM held against $2.90MM targeted at the end of January; $3.33MM against $3.34MM at
  the end of May).

## 9. Why the 3PM residual books lose after costs

- The signal is new each day. The book trades 1.29 times its gross every day.
- Commission is per share: $0.0035 is 0.32 to 0.44bp of the price for LQD, JNK, HYG and VCSH, and 1.39 to 1.66bp
  for SJNK, SPHY and BKLN, which trade near $21 to $25. Averaged over the book's trades it is 0.93bp per dollar,
  3.01% of gross a year.
- The return before costs is 0.52% of gross a year.
- The book's volatility is 0.49% a year, so (0.52 - 3.01) / 0.49 is about -5.
- At a 5-day hold the book is 0.46 before costs and -2.30 after; at 20 days 0.08 and -1.18.

## 10. Holding periods

- IG carry + roll-down and HY raw edge: the ranking of funds barely changes, and fixed weights reproduce the
  results. From 20 days up the holding period does not change the book (0.34 at 20, 30, 60 and 90 days for IG).
- NAV discount: the NAV basis has a half-life of 0.5 to 4 days. The liquid T-1 book is 0.07 after costs at 5 days,
  -0.20 at 20 and 0.13 at 60.
- Loans against HYG in the engine: 0.49, 0.32, 0.38, 0.47, 0.42, 0.33 at 1, 20, 30, 60, 90 and 120 days. The
  standard error of a Sharpe on this sample is about 0.39. A 90-day hold has about 27 non-overlapping periods.

## 11. Corrections made during this work

1. **Missing signal treated as no position.** The first paper books dropped a fund, or the whole book, when the
   spread filter left a signal missing. IG carry + roll-down then showed 0.87 on the project formula (1.17 net of
   financing) at a 1-day hold because it held nothing on 12-23 March 2020. With the signal carried forward it is
   0.22 net of financing.
2. **Previous weight kept for one fund.** The first fix kept a missing fund's old weight while re-ranking the rest,
   which unbalanced the book (treatment C in section 7). The HY raw-edge figure of 0.80 after commission came from
   this and is withdrawn. All cross-sectional rows were re-run with the convention in section 2.4.
3. **Commission on hedge legs.** Earlier tables charged commission on fund legs only. The tables here charge the
   Treasury hedge legs as well, except the 3PM residual books.
4. **Outlier days in time-series correlations.** The unclipped correlation of HYG's asset-vol z-score with the
   next-day return (0.116) comes 97% from ten days. Time-series results are reported with z-scores clipped at +/-3.

## 12. Reproduce

```bash
conda activate finm
python scripts/audit/paper_books.py      # sections 4-6 and 9  (a few minutes)
python scripts/audit/hy_edge_audit.py    # section 7
python scripts/audit/engine_runs.py      # section 8           (about 3 minutes)
```

Tables are printed and written to `results/audit/` (not committed). The time-series rows of section 4 and the
long-only reference in section 2.5 come from `notebooks/credit_etf_research.ipynb`.
