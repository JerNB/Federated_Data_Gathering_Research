# Two new tests: seasonality and two-round asking

Chinese version: [`05_refresh_and_two_round_zh.md`](05_refresh_and_two_round_zh.md).
These tests were added after the main study. They are exploratory.

## Short answer

Neither proposed idea worked.

1. **Using last year's list during gift season did not help.** In all 16 Amazon categories, the last 30 days predicted the next 30 days better than the same 30 days one year earlier.
2. **A two-round shortlist did not show a useful saving.** After correcting the estimator, it still missed all three success thresholds. At N=6,400 it often beat the best one-round accuracy, but that saving was beyond the tested range and could not be measured.

## 1. Seasonality

Question: during November-January, is last year's list better than the immediately preceding 30-day list?

Data: 16 Amazon product categories, 2,779 rolling 30-day windows.

Result: **No.** The year-ago list was worse in every category, during gift season and outside it. The gift-season difference in NDCG@100 (year-ago minus last 30 days) ranged from -0.09 to -0.50. A positive number would have supported the idea; every number was negative.

Mixing the year-ago list with the recent list helped in only 1 of 16 categories: musical instruments, both in gift season (+0.035 NDCG@100) and outside it (+0.018). It did not help in any other category.

Plain interpretation: **the product list is not repeating last year's pattern closely enough for a year-ago list to be useful.** A recent list is better, even when the calendar says the same season is coming back.

This does not prove that no seasonal effect exists. It rejects the simple rule tested here: "use last year's list in gift season."

## 2. Two-round shortlist

Question: can we save people by using the first half to find 250 candidate items, then asking the second half only about those candidates?

A fair corrected comparison used:

- the same N people in both arms;
- at most k reported choices per person;
- exact user-normalised sampling weights;
- both rounds' counts in the final two-round list.

The first run had two implementation problems: it discarded the first round's counts and did not correct the second round's sampling weights. That run is retained as `an_tworound_uncorrected_*`, but it is not used for the conclusion. The corrected run is `an_tworound_amended_*`.

Result at k=5:

- Two-round won in 42.6% of dataset/sample-size comparisons with N >= 400. The pre-registered threshold was 60%.
- Only 7 of 22 datasets had a positive gain with intervals above zero in at least half of their sample-size comparisons. The threshold was 11 of 22.
- The people-saving factor was 0.968 when endpoint-censored cells were counted at their bounds; the uncensored median was 0.856. A factor of 1.0 means no saving. A factor above 1.0 would mean fewer people. Therefore the typical result was not a saving. At N=6,400 two-round often beat the best one-round point, but the size of that saving is censored.
- The result depends on sample size and on k. At k=5, mean gain was -0.055 at N=400, -0.048 at N=800, -0.016 at N=1,600, +0.005 at N=3,200 and +0.024 at N=6,400. At k=1 the pattern is reversed: two-round wins at N=400-800 (+0.029 and +0.014 on average) and loses at N=1,600 and above.
- A larger shortlist helped: mean gain was -0.057 with 150 candidates, -0.020 with 250, and +0.007 with 500. This is a small gain, not a demonstrated saving.

Plain interpretation: **at k=5 and k=20 with N of 200 or more, splitting the people into discovery and confirmation wastes information at small and medium N; at the largest N the design may help a little. At k=1 the effect flips with N. This experiment does not show a useful reduction in the number of people at any k.**

Do not generalise this to every possible two-round protocol. The test used an equal split, a fixed shortlist, and equal-weight pooling.

## Files

- Season script: `scripts/ssp/an_season.py`
- Season windows and verdict: `results/small_sample_popularity/03_tables/an_season_windows.csv`, `an_season_verdicts.json`
- Two-round script: `scripts/ssp/an_tworound.py`
- Corrected two-round outputs: `results/small_sample_popularity/03_tables/an_tworound_amended_cells.csv`, `an_tworound_amended_summary_cells.csv`, `an_tworound_amended_verdicts.json`
- Figures: `results/small_sample_popularity/04_figures/an_season.png`, `an_tworound_amended.png`
- Added Amazon categories: 11; raw downloads are ignored and are not committed.

## Verification

The season-window checks found no future leakage or metric-range errors. The corrected two-round estimator matched hand-computed user-normalised counts and was unbiased in a 40,000-repeat Monte Carlo probe, and throwaway reruns reproduced the first dataset's rows exactly for both experiments. Reviewers (same provider, not an outside audit) found and I fixed the two estimator defects above and several wording problems. No repository test suite exists.
