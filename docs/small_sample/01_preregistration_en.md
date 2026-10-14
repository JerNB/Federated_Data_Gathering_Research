# Pre-registration: Small-Sample Popularity Estimate

Written 2026-10-14, before the overnight run. Chinese version: [`01_preregistration_zh.md`](01_preregistration_zh.md).
Raw results will appear in `results/small_sample_popularity/`; the write-up in `02_results_en.md`.

## 1. The question, in plain words

Take a small group of people from a larger community, keep a few of their past choices, and use only that
to build a ranked list of items. **How good is that list, and when is a personalized recommender trained on
the same small sample worth more than just showing the popularity list?**

The popularity list is treated like a statistic (a mean, a variance), except the thing being estimated is a
ranking.

## 2. Two kinds of "truth" (both are reported, side by side)

| Name | What it is | Question it answers |
| --- | --- | --- |
| **Truth A** | The item counts over *everyone* in the community, in the same time window as the sample | Estimation: how close is the small-sample list to the full-community list? |
| **Truth B** | What the community actually chose *later* (after a cut-off time) | Prediction: how useful is the list for what people do next? |

Other words used below:

- **N** = number of people sampled; **k** = number of past choices kept per person ("all" = every earlier choice).
- **Draw** = one random sample. Every cell below is repeated many times so the noise can be measured.
- **Panel** = 1,500 (or fewer) held-out people who are *not* in any sample. They are shown k of their own past choices and we score the list on their real later choices.
- **Stale list** = the community's item counts from an earlier period (last year's list).

## 3. Data (11 public datasets)

All are public downloads; none are stored in Git. Recreate with `datasets/fetch.sh`, then `scripts/ssp/prep_data.py --all`.

| Dataset | Domain | People in the group | Items seen | Top item's share | Zipf slope s |
| --- | --- | ---: | ---: | ---: | ---: |
| ml-latest (MovieLens, 2023 snapshot) | movies | 104,860 | 32,014 | 48.9% | 0.62 |
| ml-1m (MovieLens 1M) | movies | 3,608 | 3,434 | 48.2% | 0.59 |
| amazon-video-games | products | 8,844 | 17,019 | 5.3% | 0.49 |
| amazon-musical-instruments | products | 6,486 | 18,795 | 8.2% | 0.58 |
| amazon-software | products | 11,610 | 12,725 | 15.2% | 0.69 |
| amazon-movies-tv | products | 26,300 | 137,876 | 3.5% | 0.38 |
| amazon-books | products | 38,564 | 359,626 | 1.8% | 0.39 |
| gowalla | places (check-ins) | 30,180 | 878,456 | 5.0% | 0.50 |
| brightkite | places (check-ins) | 7,936 | 430,185 | 5.4% | 0.60 |
| kuairand-pure | short videos (clicks) | 7,993 | 6,583 | 39.0% | 0.64 |
| lastfm-1k | music (artists) | 712 | 128,151 | 70.2% | 0.39 |

Rules fixed now:

- A *choice* is a rating of at least 4 (explicit data) or any recorded interaction (implicit data). Only the first time a person touches an item counts.
- Cut-off time T0 = the 80th percentile of all choice times. Earlier = *collection*; later = *future*. Stale list = choices before the 55th percentile.
- Only people with at least 20 earlier choices (10 for the Amazon sets, 5 for lastfm-1k) form the community. 80% of them are the **group** being sampled from; 20% are held out for the panel.
- N never exceeds 40% of the group, so the finite-population effect stays small.

## 4. What is sampled and estimated

| Grid | Values |
| --- | --- |
| N (people) | 20, 40, 100, 200, 400, 1000, 2000, 4000, 10000, 20000 (capped at 40% of the group) |
| k (choices each) | 1, 2, 5, 10, 20, all (not above the dataset's minimum) |
| Draws per cell | 60 for popularity lists, 20 when models are trained, 30 for rank intervals, 40 for subgroups, 20 for privacy |

Estimators:

| Name | What it does |
| --- | --- |
| `count` | Rank items by how many sampled people chose them (the plain popularity list) |
| `shrunk` | Blend the sample counts with the stale list; blend strength estimated from the sample |
| `lcb` | Rank by a cautious lower bound on each item's rate |
| `ht` | Counts re-weighted to undo a biased sampler (only for biased samplers) |
| `knn`, `ease` | Personalized models trained on the sample only (item neighbours; EASE) |
| `hybrid` | `(1-w)·popularity + w·personalized`; w and the model are chosen **inside the sample** by hiding part of each sampled person's choices. Two ways to hide: *random* and *latest-first (time order)* |

Metrics: overlap of the top 10 / top 100 with the truth; rank-biased overlap (0.9 and 0.98); Kendall tau on the true top 100; NDCG@100 of the list under the truth's counts (a soft score that does not need exact order); and for the panel, NDCG@100 and recall@100 on the people's real later choices.

## 5. What I already know before running (full disclosure)

- Pilot, about 20 draws on ml-1m only: the code runs, the full sample reproduces Truth A exactly, and a few cells looked like this:
  the stale list beat every small sample; personalized models lost to popularity at k=10 but won at k=20, N=1000; hybrids that picked their weight by *random* hiding trusted personalization more than the time-ordered ones did.
  **These pilot draws are not used in any claim**; the real runs use different seeds and all 11 datasets.
- Theory check (no sampling): for every dataset, the number of correctly **ordered** items at the top of the list is only 1 to 2, even with N = 5,000 people (Dyer-Owen bound). Identifying the exact top 10 needs roughly 10^4 to 10^7 people. So claims below use soft scores (NDCG, mass captured) and treat exact overlap as descriptive.
- The distance between Truth A and Truth B is large on the reference lists (for example the "perfect later list" scores 0.246 vs 0.184 for the full-community earlier list on ml-1m's panel).

## 6. Confirmatory claims (3)

The test rules below are fixed. Each is judged per dataset, then across datasets by a sign count and a pooled interval. Intervals are bootstrap over draws, 95%. No tuning after seeing results.

### C1. Breadth beats depth for estimating the list

*Plain:* at the same total number of kept choices, many people with a few choices each give a better popularity list than few people with many.

- **Quantity:** `count` NDCG@100 under Truth A. For each dataset and each budget B = N·k present twice or more in the grid (B = 200, 400, 1000, 2000, 4000, 10000…), difference = (widest cell: largest N, smallest k) minus (deepest cell: smallest N, largest k).
- **Prediction:** difference > 0 in at least 75% of (dataset, budget) pairs, and the pooled interval excludes zero.
- **Falsifier:** at most 50% of pairs positive, or the pooled interval is entirely below zero.
- **Why we expect it:** choices from one person are alike, so extra choices from the same person add less than a new person does (the "design effect").

### C2. A hybrid with honest weights is never worse than the better pure list, and sometimes better than both

*Plain:* mixing popularity and personalization, choosing the mix using only the sample, matches the better of the two everywhere and beats both in a good share of cells.

- **Quantity:** panel NDCG@100 in the model-training stage (7 datasets, N from 100 to 4,000, k = 5, 10, 20, all). `hybrid_time` (weight chosen by hiding each sampled person's latest choices) versus the better of `popularity` and `personalized_time` (best pure model chosen the same way).
- **Prediction:** (a) in at least 90% of cells the hybrid is no more than 0.002 below the better pure list; (b) in at least 20% of cells it is above both, paired interval over draws excludes zero; (c) `hybrid_random` (random hiding) puts at least as much weight on personalization as `hybrid_time` in at least 70% of cells.
- **Falsifier:** (a) fails in more than 25% of cells, or (b) holds in fewer than 10%.

### C3. A crossover sample size exists, and shrinks as k grows

*Plain:* with enough people, a model trained on the sample beats the popularity list on real later choices; the more choices each person gives, the fewer people are needed.

- **Quantity:** N*(k) = the smallest N in the grid where `personalized_time` beats `popularity` on the panel, with a 95% interval above zero.
- **Prediction:** for k = 20 and k = all, N*(k) exists (≤ 4,000) in at least 5 of 7 datasets; where it exists for several k, N*(k) does not increase with k.
- **Falsifier:** at k = all, N* does not exist in at least 4 of 7 datasets. That would not be a failure of the project: it would mean "popularity wins at every sample size tested", reported as a lower bound.

## 7. Exploratory hypotheses (stated now, labelled exploratory)

Each carries a prediction so that a miss is visible. False-discovery control: Benjamini-Hochberg, q = 0.10, within each family.

| # | Hypothesis | Prediction |
| --- | --- | --- |
| X1 | The stale list beats a small fresh sample until N reaches a large share of the group | Crossover N_s ≥ 30% of the group in most datasets |
| X2 | Shrinking toward the stale list helps most near the crossover; shrinking toward a flat (item-independent) prior changes nothing | Flat-prior ranks are identical to raw counts (exact control); gain peaks near N* |
| X3 | Error falls as a power law in N with a floor; the exponent is ordered by how heavy the head is (Zipf slope); head-order metrics behave like a threshold, not a power law | Rank correlation of exponent and slope above 0.6 across the 11 datasets |
| X4 | The Dyer-Owen bound is conservative | Observed correctly ordered depth ≥ bound at every N |
| X5 | Naive bootstrap rank intervals under-cover in the crowded middle of the list; the Mogstad et al. sets reach nominal coverage but are wide | Naive coverage below 0.93 in the ranks 51-200 band in most cells; Mogstad at least 0.95 |
| X6 | Under activity-biased sampling, the uncorrected count beats the weighted (Horvitz-Thompson) count at small N, and loses at large N | A crossover N exists in most datasets |
| X7 | Sampling recent choices wins for Truth B, and loses for Truth A (a sign flip) | Opposite signs in most datasets |
| X8 | For a real subgroup (genre fans, a join period, activity tercile), a small sample plus the global list beats either alone; the sample size where the subgroup sample alone wins grows with how different the subgroup is | Positive rank correlation between distinctiveness and that sample size |
| X9 | Privacy noise costs mostly at the head of the list; central DP at ε ≥ 2 is nearly free once N is a few thousand, local DP needs ~100x more people | Statements in sample size ratios, per dataset |
| X10 | The gap between Truth A and Truth B (drift) is larger than the sampling error at any N below 40% of the group | A-oracle list vs B-oracle list gap exceeds sampling gap in most datasets |
| X11 | Surprise hunt: scan every (dataset, N, k, estimator, metric) curve for non-monotone shapes and sign reversals; each hit is re-run on fresh seeds | Reported as "found / replicated / not replicated" counts |

## 8. Statistical rules

- Cells are compared with differences of per-draw means; intervals are a cluster bootstrap over draws (2,000 resamples). Panel comparisons are paired on the same draw.
- Across datasets: report the count of datasets with a positive effect, and a pooled estimate with an interval over datasets. No p-value is used to rescue a failed claim.
- Anything exploratory is labelled "exploratory" wherever it appears and carries a replication status.
- Ties in rankings are broken randomly; truth lists break ties by item index.
- Results that disagree with the predictions above are reported at the same prominence as those that agree.

## 9. What this study cannot say

- Samples here are random draws from a community; they are not a real small group, except in the subgroup stage (X8).
- Public datasets are volunteer or platform logs; trends are "in these data", not universal.
- Federation is simulated: clients are labels on user ids; there are no devices, dropouts or real secure aggregation. The privacy stage adds noise to counts only.
- Personalized models here are trained on at most about 1,500 candidate items (to keep the machine quiet); this favours popularity at large N.

## 10. Compute rules for the overnight run

CPU only, no GPU. A governor reads die temperature, fan speed, memory and disk every 3 seconds. On this machine the fans idle at about 1,400 rpm and start rising once the die stays near 80 °C, so the run slows or freezes when the die passes 72 °C or the fans pass 1,750 rpm, and freezes all workers when it passes 78 °C or 2,100 rpm. At most 4 workers, 8 GB of worker memory, and at least 10 GB of disk left free. Every task is repeatable from its key.

## 11. Amendments (2026-10-14, made after an independent code review and BEFORE any real run)

An independent review of the code (a subagent on the same provider, so not an outside audit) found issues. Each is fixed or declared here. No real-run result had been seen; the pilot draws on ml-1m are not used anywhere.

| # | Change | Why |
| --- | --- | --- |
| A1 | The **stale list** (choices before the 55th percentile) is a nested part of Truth A (choices before T0), about two thirds of its rows, so "stale beats samples" would be partly mechanical. X1 and X2 are therefore scored against a **window truth** (choices between the 55th percentile and T0 only), using samples drawn only from that window (`recent_window` arm). The main grid's `shrunk` estimator is still reported but is flagged "prior overlaps the truth". | Remove overlap between prior and truth |
| A2 | C3 uses **k = 10** and **k = all** (k = 20 exists in only 4 of the 7 model-stage datasets, so "5 of 7" could never be met). Prediction: N* exists at k = all in at least 5 of 7 datasets and at k = 10 in at least 4 of 7. Falsifier unchanged. | Rule was unmeetable by construction |
| A3 | Frame rule: at least 20 earlier choices when 3,000 or more people qualify, else 10, else 5. In practice amazon-movies-tv and amazon-books use 20; the three smaller Amazon sets use 10. | Table in section 3 was wrong |
| A4 | Biased samplers (activity-weighted, inverse-activity) use **Poisson sampling**: each person joins independently with a known probability, so the sample size is random (about N) and the weighted count is exactly unbiased. | The first design had biased weights at large N |
| A5 | Ties in the panel scoring are broken at random. Time-ordered hiding cannot separate choices with the same timestamp (common on ml-1m and bulk-imported data), so there it is only partly time-ordered. The ml-1m panel has 194 people only, so its panel numbers are noisy. | Reported limits |
| A6 | Genre subgroups are the 15% of people with the highest share of that genre; the sample size N is capped at 40% of the subgroup. | Keep a real sampling situation |
| A7 | `ht` (weighted count) is reported only for the activity-biased samplers, not for "recent rows". | It is not a valid weighting there |
