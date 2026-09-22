# Analysis Spec — Data Quality & Time-Series Diagnostics

**Status:** Draft, in progress
**Scope:** `account_balances_daily_RAW.csv`, cross-referenced with `accounts.csv`
**Primary artifact:** `data_quality_diagnostics.ipynb`
**Also touched:** `flawed_model_squad_draft.ipynb` (two exploratory cells added: balance/inflow/outflow over time, day-over-day anomaly flagging — squad's original forecast/rebalancing cells untouched)
**Not covered here:** `transfers_log_RAW.csv`, the rebalancing rule itself, Part 2 forecast design

## 1. Objective

Quantify what's actually wrong with the raw balance data before touching the squad's
forecast/rebalancing logic (Part 1), and characterize the time-series structure (trend,
outliers, weekly/monthly seasonality) needed to design a better forecasting approach (Part 2).

## 2. Environment

None of the following were in `requirements.txt` before this work; added as needed:

| Package | Version | Used for |
|---|---|---|
| `matplotlib` | 3.11.2 | static small-multiples plots |
| `plotly` | 7.1.0 | interactive daily-detail charts |
| `statsmodels` | 0.15.0 | STL/MSTL seasonal decomposition |
| `scipy`, `patsy` | 1.18.1, 1.0.3 | statsmodels dependencies |

## 3. Data quality findings (`data_quality_diagnostics.ipynb`, Sections 1–6)

### 3.1 Currency labels
- **Finding:** 3 real currencies (COP, USD, MXN) written 9 different ways (`cop`,
  `Colombian Peso`, `USD`, `US Dollar`, ...).
- **Decision:** normalize via an explicit lowercase map → `currency_norm`. No FX conversion
  applied — every series stays in its native currency throughout.
- **Impact if unfixed:** any `groupby("currency")` silently splits one currency into up to 3
  buckets.

### 3.2 Date parsing
- **Finding:** the squad's `pd.to_datetime(errors="coerce")` + `dropna()` drops **293/1,668
  rows (17.6%)** — every non-ISO date, not "a few" as the original code comment claims.
- **Root cause identified:** the separator is unambiguous — every `/`-formatted date is
  `DD/MM/YYYY`, every `-`-formatted (2-2-4) date is `MM-DD-YYYY`. Verified with **zero
  exceptions across all 1,668 rows**, including the "ambiguous" (both parts ≤12) cases that
  currency/account grouping could not resolve on its own.
- **Decision:** `parse_date_safely()` dispatches on the regex-matched separator/format instead
  of relying on `pd.to_datetime`'s built-in inference. Recovers 100% of rows (0 unparseable).
- **Downstream effect (Section 7):** the naive parsing doesn't just drop rows — it introduces
  **fake multi-day gaps** (190 rows with a 2-day gap, 38 with a 3-day gap, 5 with a 4-day gap)
  into what should be, and after the fix is, a perfect 1-row-per-calendar-day series for every
  account (270/270 days, 0 missing calendar days). Any day-over-day diff computed on the naive
  series was sometimes comparing balances up to 4 calendar days apart.
- **Concrete consequence:** the negative-balance row for ACC-003 on `24/06/2025` (-2.89B COP)
  is silently dropped by the naive parser (24 is invalid as a month) — a second reserve-account
  shortfall hidden by the parsing bug alone, in addition to the one on 2025-04-03.

### 3.3 Duplicate (account_id, date) rows
- **Finding:** 96 rows in 48 conflicting pairs — same account, same day, two different
  balances (not exact duplicates; `drop_duplicates()` won't catch them). Differences are small
  (~0.06%–0.1%) but real.
- **Decision:** **flagged only, not resolved.** The correct resolution (keep-latest / average /
  investigate-source) depends on which upstream system re-wrote the row — out of scope for a
  data-quality pass. Downstream (Sections 7–10), duplicates are resolved with a simple mean for
  plotting purposes only — explicitly called out in the notebook as not the real resolution
  rule.

### 3.4 Missing values
- **Finding:** `balance` 68/1,668 (4.1%), `inflow` 65 (3.9%), `outflow` 65 (3.9%). Spread
  across all 6 accounts; no single account or date window absorbs them. 0 rows missing all
  three at once.
- **Validation performed:** checked the accounting identity
  `balance_t = balance_(t-1) + inflow_t - outflow_t` on fully-populated rows — holds within ±1
  unit on **92.1%** of rows, but the residual has a fat tail (max ≈8.14B), which lines up with
  rows already flagged as anomalies (Section 8) — i.e. the identity is reliable enough to
  consider for imputation, but not blindly.
- **Decision:** **not yet imputed.** Flagged as needing an explicit choice before any forecast
  consumes these columns.

### 3.5 Negative balances & rule coverage gap
- **Finding:** only 4 negative-balance rows in the whole dataset:
  - ACC-005, 2025-04-10, -161.9M MXN
  - ACC-001, 2025-07-18, -2.616B COP
  - ACC-003, 24/06/2025, -2.891B COP
  - ACC-003, 2025-04-03, -4.071B COP
- **Cross-check:** 3 of 4 belong to ACC-003 and ACC-005, which are exactly the two
  `reserve`-type accounts **absent from the squad's `THRESHOLD` dict** (which only covers
  ACC-001, ACC-002, ACC-004, ACC-006).
- **Conclusion:** this is a **rule-coverage gap**, not sensor/data noise — the rebalancing rule
  has zero visibility into the accounts that actually went negative.

## 4. Outlier detection (Section 8)

- **Method:** per-account, robust (median/MAD) modified z-score, threshold `|z| > 3.5`.
- **Key methodological decision:** `balance` is tested on its **day-over-day change**, not its
  raw level. Several accounts trend over the 9-month window; a level-based check flags the
  entire trending region as anomalous (verified empirically — a level-based pass on ACC-004
  falsely flagged 25 consecutive early-January rows that are just its normal decline).
  `inflow`/`outflow` don't carry that trend and are tested on the raw daily value instead.
- **Result:** 59 points flagged across 6 accounts × 3 series. Full list in `outlier_summary`.
- **Currency-grouped view:** accounts sharing a currency (COP: ACC-001/003, USD: ACC-002/006,
  MXN: ACC-004/005) are overlaid on one shared axis — same unit, so legitimate, not a
  dual-axis chart. Pattern observed: a single-day spike-then-revert anomaly shows up in
  **every account**, and within each currency pair the largest-magnitude spike consistently
  lands on the `reserve` account, not the `operational` one.

## 5. Seasonal decomposition (Sections 9–10)

- **Method:** MSTL (`periods=[7, 30]`, `robust=True`) per account per series. Robust mode
  down-weights the Section-8 outliers so they don't distort the trend/seasonal split. Small
  missing-value gaps are linearly interpolated first — for decomposition only, not the
  Section 3.4 imputation decision.
- **Caveat (explicit in the notebook):** the "30-day" component is a fixed cycle counted from
  2025-01-01, not aligned to calendar months (28–31 days) — read it as "a recurring ~monthly
  rhythm," not "end-of-month effect."
- **Findings:**
  - `balance`: ~0 seasonal strength (weekly and monthly) on every account — expected, it's a
    stock, not a flow.
  - Weekly: strongest in the two reserve accounts — ACC-003 (`outflow` 0.581, `inflow` 0.502),
    ACC-005 (`outflow` 0.467, `inflow` 0.411). Day-of-week shape for the strongest case
    (ACC-003 inflow): weekdays average roughly +9M to +26M COP, weekends roughly -42M to -52M
    COP — a clean weekday-activity / weekend-quiet pattern.
  - Monthly: weaker overall, but **highest for ACC-004** (`outflow` 0.306, `inflow` 0.202) —
    the same account the squad's original notebook flagged as producing unreliable forecasts.
    The shape is real but not a single clean peak — a plausible contributor to that
    instability, not the whole story.
- **Interactive views:** Plotly small-multiples for daily detail (Section 9) and for both
  seasonal components (Section 10), plus full 5-panel breakdowns
  (observed/trend/weekly/monthly/residual) for the strongest weekly case and the strongest
  monthly case.

## 6. Open items (not resolved by this work)

1. Duplicate (account_id, date) resolution rule — needs a decision, not just detection.
2. Missing balance/inflow/outflow — needs an explicit imputation or exclusion policy.
3. Currency normalization is label-only — no FX conversion; any cross-currency aggregation
   still needs an FX rate decision.
4. `transfers_log_RAW.csv` not yet cross-referenced against the balance-change series (would
   help confirm whether flagged "anomalies" correspond to real recorded transfers vs. data
   errors).

## 7. Implications for Part 1 / Part 2

- **Part 1 fix list**, all quantified above with before/after counts: currency normalization,
  date reparsing (separator rule), duplicate flagging, negative-balance/threshold-coverage gap.
- **Part 2 forecast design** should not reuse a flat 14-day trailing average on `balance`
  directly: the level carries no exploitable weekly/monthly seasonality (it's a stock), but the
  **flows** that drive it do — a forecast built on `inflow`/`outflow` seasonality (weekly for
  reserve accounts, ~monthly for ACC-004) is better supported by the data than smoothing the
  balance level.
