# Gaps in the supplied corporate-action file

Found 2026-11-16 on `etf_dividends_splits.parquet`. Everything below is reproducible with
`src/data_quality.py` (`DataLoader("data").load().quality.summary()`).

## 1. Missing cash-distribution records, Dec 2024 - Jul 2025

Number of universe ETFs with at least one cash distribution going ex in the month:

| Month | ETFs with a record | Same month a year earlier | ETFs that paid a year earlier and have no record |
|---|---|---|---|
| 2024-11 | 205 | 205 | not a gap month |
| 2024-12 | 178 | 212 | 35 |
| 2025-01 | 1 | 35 | 34 |
| 2025-02 | 41 | 197 | 156 |
| 2025-03 | 42 | 197 | 156 |
| 2025-04 | 42 | 208 | 166 |
| 2025-05 | 42 | 203 | 161 |
| 2025-06 | 42 | 205 | 163 |
| 2025-07 | 42 | 208 | 166 |
| 2025-08 | 205 | 205 | not a gap month |

- 167 of the 213 ETFs are missing at least one month: 120 miss six months, 31 miss all eight.
- The price still drops on the ex-date. Example, BIL: -40bp on 2024-12-19 and -33bp on 2025-02-03, 2025-04-01
  and 2025-06-02, each with no record, against a recorded monthly distribution of about the same size in other months.
- The 168 rows typed `Omitted` or `Discontinued` are not the cause: they carry no amount and the median
  price move on their dates is zero.
- Recorded distributions against the `yield` field, median over funds, in percentage points of shortfall:
  clean funds 1.44 (2023), 0.77 (2024), 0.42 (2025); excluded funds 1.30, 1.04, 2.69.

## 2. Reverse splits that are not recorded

| ETF | Date | Close before | Close after | Ratio |
|---|---|---|---|---|
| BBAG | 2021-04-12 | 26.96 | 53.90 | 1.999 |
| USFR | 2022-03-24 | 25.15 | 50.29 | 2.000 |

No other overnight move in the universe is within 3% of a factor of 2, 3, 4, 5 or 10 (or its inverse) without a
split record. The nine recorded splits (PREF, HYLB, HYMB and six Schwab funds) reconcile: total return on each
split date is between 0.00% and 0.56%.

## 3. Effect on a backtest that uses the file as supplied

Engine runs with the exclusion switched off (`BacktestConfig(clean_universe_only=False)`), $10MM positions:

| Run | Effect |
|---|---|
| Hold USFR | +$10,023,921 on 2022-03-24 (the largest PnL on any other day is $71,770) |
| Hold BIL | -$40,461 on 2024-12-19; -$32,806 on each of 2025-02-03, 2025-04-01, 2025-06-02; distributions received $207,966 in 2025 against $460,796 in 2024 |
| Hold SHV (complete records) | worst day -$8,146; distributions $407,642 in 2025 against $500,775 in 2024 |
| `equal_weight_long` on all ETFs | distributions $281,016 in Jan-Jul 2025 against $1,307,771 in Jan-Jul 2024 |
| `equal_weight_long` on the 46 clean ETFs | distributions $1,301,244 in Jan-Jul 2025 against $1,301,685 in Jan-Jul 2024 |

BBAG has no 3PM snapshot until 2023-02-01, so it could not be held on its split date.

## 4. Treatment in this branch

ETFs with a missing month or an unrecorded split are excluded from trading for the whole sample
(`clean_universe_only=True`, the default). The remaining 46 ETFs:

| Group | ETFs |
|---|---|
| IG corporate (13) | IGSB, SCHJ, SPSB, VCSH, IGIB, SCHI, SPIB, VCIT, IGLB, SPLB, VCLT, LQD, USIG |
| IG floating rate (3) | FLOT, FLRN, FLTR |
| High yield (7) | HYG, JNK, USHY, SPHY, SHYG, SJNK, EUHY |
| Bank loans (2) | BKLN, SRLN |
| Treasuries (9) | SHV, SHY, SCHO, IEI, SCHR, IEF, TLH, TLT, GOVT |
| Other (12) | AGG, BND, BOND, IUSB, SCHZ, EMB, EMHY, HYMB, IAGG, IGOV, ISHG, SCHP |

No preferred-stock ETF and no target-maturity ETF is in the clean set. The excluded 167 are to be brought
back in a separate track that imputes the missing distributions and applies the two splits, documented on its own.

## 5. Limits of the rule

- The exclusion list is built from the full sample, so it is not a point-in-time filter. It is a filter on the
  integrity of the supplied records, not on performance.
- A fund is flagged only if it paid in the same month a year earlier. A fund that pays irregularly could have a
  missing record the rule cannot see. IGOV is in the clean set because it had no distribution in any of the
  reference months.
- Check on the clean set: no clean fund has fewer records in any gap month than in the same month a year earlier.
  A screen for ex-date-sized drops with no record on mass ex-dates in the gap window finds 26 fund-days among the
  clean funds (818 among the excluded funds). Of the 26, 17 are on 2024-12-18, when the median clean fund
  returned -78bp; the other 9 are in EUHY, IGOV, IAGG and BKLN on dates when the fund has no distribution record
  within three days in either year.

## 6. Questions for the instructor

1. Is a corrected dividend file available for Dec 2024 - Jul 2025, and for the BBAG and USFR reverse splits?
2. Is excluding the affected ETFs for the whole sample acceptable, given that universe filters are meant to use
   only data available when applied?
