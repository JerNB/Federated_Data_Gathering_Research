# Results: Small-Sample Popularity Estimate

Chinese version: [`02_results_zh.md`](02_results_zh.md). Pre-registered plan: [`01_preregistration_en.md`](01_preregistration_en.md).
All numbers below come from `results/small_sample_popularity/03_tables/`; every figure is in `04_figures/`.
Run on 11 public datasets, 117,069 small tasks plus 1,220 fresh-seed replications, overnight on one laptop with the fans kept at idle.
Later work (docs 03–23: weekly memory, self-chosen memory, volunteers, privacy, and a review package for outside readers) is indexed in [`00_start_here_en.md`](00_start_here_en.md) section 7; this page covers the main study only.

## 1. The answer in one paragraph

A popularity list built from a few hundred people is already close to the full-community list in soft quality (NDCG@100
about 0.88 at N = 400), although the exact top 10 is still far from settled. **The bigger gap is time.** Measured against
what the community chooses *next*, the distance between its earlier list and its later list is far larger than the
sampling gap, in 10 of 10 datasets. The recency-aware methods we tried recover only a small part of that distance
(a median +0.013 of a 0.66 gap on the list; +0.002 for held-out people). A personalized model trained on the same
sample is still improving at the largest tested sample size (up to 4,000 people), while the popularity list is within
5% of its best much earlier; in a post-hoc small-N stage it beats the popularity list from as few as 20 people in 3 of
7 datasets. **Our three pre-registered predictions did not hold as written**: C1 inconclusive, C2 falsified, C3 mixed.
The reasons are in section 2, together with the checks we ran on our own first explanations.

## 2. Scorecard of the three pre-registered claims

| Claim | Prediction | Verdict | What the data say |
| --- | --- | --- | --- |
| C1 Breadth beats depth | At equal total choices, many people with few choices each beat few people with many | **Inconclusive** | 61% of 46 (dataset, budget) pairs favour breadth; pooled difference +0.0005 NDCG@100, interval [-0.003, +0.004]. For plain counts breadth and depth are about equal in quality. For the statistically unbiased weighted estimator **depth wins** (-0.008, interval [-0.012, -0.005]). |
| C2 Honest hybrid | A blend chosen inside the sample is never worse than the better pure list, and sometimes better than both | **Falsified** (two registered falsifiers triggered) | The hybrid is within 0.002 of the better pure list in only 70% of 132 cells (falsified below 75%); on average it is *worse* by 0.0018 (interval [-0.0029, -0.0008]) and better in 1 of 7 datasets. It beats both in 25% of cells (needed 20%: met). Random hiding did *not* trust personalization more than time-ordered hiding (38% of the 78 eligible cells; falsified at 50% or below). |
| C3 Crossover size | A sample size N* exists where a personalized model beats popularity, and it shrinks as histories grow | **Mixed** | At k = all, N* exists in 5 of 7 datasets (needed 5: met); at k = 10 in 3 (needed 4: missed). No dataset has N* rising with k. N* at k = all is at or below the smallest registered size (100) in 4 datasets; the post-hoc small-N stage puts it at or below 20 people in 3. |

What we checked about our own first explanations (`03_tables/an_followups_verdicts.json`):

- **C1.** "Extra choices from the same person carry as much information as choices from a new one" only restates the
  result. What we can show is that breadth makes lists more *repeatable*: the draw-to-draw standard error is smaller
  for the wide cell in 39 of 46 pairs (median ratio 0.88), while average quality is the same. The weighted estimator's
  loss at small k is most likely weight variance (each choice is weighted by history length / k), not information.
- **C2.** The in-sample hold-out that picks the blend has almost no skill beyond the base rate: personalizing beats
  popularity on the panel in 33% of draws, so always guessing "popularity" is right 67% of the time; the time-ordered
  hold-out is right 68%. It also overstates the benefit of personalizing (by +0.014 for time-ordered hiding and +0.017
  for random hiding), which fits the hybrid being worse on average.

## 3. Findings worth building a paper on (all exploratory unless stated)

1. **Drift is bigger than sampling error (X10, 10 of 10 datasets).** Against an oracle of the future, the gap between
   "the community's earlier list" and "its later list" is far larger than the gap between a 1,000-person sample and the
   full list (median 0.021 vs 0.000 panel NDCG@100; against the truth lists 0.66 vs 0.003). The panel number for
   sampling is near zero partly because the panel metric is insensitive; read it together with the list-level number.
   Figure: `an_explore_a_x10_drift_vs_sampling.png`.
2. **Recency flips the sign of the target (X7).** A recent-window sample predicts the community's later list better
   than a uniform sample of the same size (median +0.037 NDCG@100, in 10 of 11 datasets) and is worse against the
   full earlier list (-0.163, expected by construction). For held-out people the gain is small: median +0.002, negative
   in gowalla and lastfm-1k. Recent rows only: 8 of 11 datasets flip. Figure: `an_explore_b_x7_recency.png`.
3. **Recency-weighted counts help (X12).** They beat plain counts on later choices in 179 of 223 cells and lose in 2
   (median +0.013 NDCG@100 on the list; the gain on held-out people is consistent but small, at most about +0.006).
   On the score against the same recent window it loses in 119 of 223 cells, so it trades one target for the other.
   Figure: `an_window_x12_decay_minus_count_kall.png`.
4. **Two speeds of learning (X17).** From N = 100 to the largest tested N the popularity list's panel score grows by a
   median factor 1.22, the best personalized model's by 1.94 (the personalized curves start low, which inflates the
   ratio; the curves in the figure are each scaled to their own value at the largest N). Popularity reaches 95% of its
   best at least two grid steps earlier in 4 of 7 datasets; nothing is observed beyond N = 4,000. Figure: `an_curves_two_speeds.png`.
5. **The crossover is lower than the registered grid could see (C3 companions, post hoc).** The small-N stage was
   added after the registered grid showed left-censoring, and the baseline there is a popularity list from the same few
   people. At k = all, personalizing beats
   popularity at N <= 20 in ml-latest, amazon-software and kuairand-pure, at 100 in ml-1m and at 1,000 in gowalla; it
   never does up to 2,000 in the two smaller Amazon sets. Counting only items a model can reach, amazon-video-games
   joins at 2,000. Figure: `an_c3_checks_small_n.png`.
6. **Unbiased is not better (X15).** The statistically unbiased weighted count never beats the plain count, or only at
   N >= 10% of the group, in 47 of 50 dataset-k series: the textbook bias-variance trade-off, measured.
7. **Rank uncertainty (X5).** A naive bootstrap rank interval covers only 0.85 of true ranks 51-200 (median) and falls
   below 0.93 in 79% of cells; the Mogstad et al. sets reach 0.95 in every cell but are 3.0 times wider.
   Figures: `an_explore_b_x5_coverage.png`, `an_explore_b_x5_widths.png`.
8. **Privacy (X9, mixed).** Central DP at epsilon >= 2 and N >= 2,000 keeps a median 0.86 of the non-private quality;
   local DP needs a median 50 times more people (a lower bound). Noise removes 96% of top-10 overlap but 93% of
   top-100 overlap: the damage is not concentrated at the head as predicted.
9. **Subgroups go the other way (X8, prediction failed).** The more a real subgroup differs from the global list, the
   *smaller* the sample at which its own sample beats the global list (Spearman -0.44, interval [-0.56, -0.31]).
   Our pre-registered direction was wrong. Part of the number is censoring: the 70 series that never reached a crossover
   are the less distinctive ones. Among the 94 series that did, the correlation is -0.20 (interval [-0.37, -0.01]), still
   negative but weak. The second half of X8 also failed: the shrunk list was at least as good as both pure lists at every
   N in only 7% of series.
10. **A sample cannot rank its own draws (X14, mixed).** The bootstrap predicts the level of a list's quality well
    (median predicted/real ratio 1.00 at N >= 200), but across draws predicted and real quality correlate at only 0.18
    (median Spearman). Note that NDCG is close to 1 here, so the ratio check is weak evidence by itself.
11. **Scaling law across datasets (X3, mixed).** The fitted error exponent is rank-correlated with how top-heavy a
    dataset is (Zipf slope): 0.54 at k = all and 0.76 at k = 5 over 11 datasets (we predicted above 0.6). A power law
    beat an exponential alternative in only 4 of 11 datasets at k = all and 0 of 11 at k = 5, so "one power law for all
    metrics" is too simple. Figure: `an_explore_a_x3_exponent_vs_zipf.png`.
12. **Shrinking toward an older list (X2, not supported).** Blending a fresh sample with the older list gains up to
    0.76 NDCG@100 on the window truth, but the gain peaks at the smallest samples (median N = 20), not near the
    crossover as predicted. Figure: `an_window_x2_window_shrinkage_gain.png`.

## 4. What the surprise hunt found

We scanned 40,356 comparisons (non-monotone curves, order reversals, interactions, cross-dataset sign differences,
privacy noise "helping") with false-discovery control. In the four classes we could replicate, 443, 325, 1,802 and 189
comparisons passed it (two more classes, with 253 and 27 hits, could not be re-run because the fresh data lacked their
cells, so they are not findings). We re-ran the 50 largest eligible ones (effect at least 0.01, not relying on the main-grid `shrunk` estimator) on fresh
random seeds: **all 50 replicated** (same direction and p < 0.05 on fresh seeds, on the same data and the same panel, so this tests
draw noise only). Replication here means "stable property of these data", not "important": most are
small-N artifacts (for example, a naive linear blend of sample shares and global shares lets noisy sampled items
outrank globally popular ones at N = 20) and breadth-depth interactions on the check-in datasets. Two classes (valleys
in the hybrid weight and privacy noise helping) found nothing. Details: `03_tables/hunt_candidates.csv`.

## 5. A suggested framing

> *Small-sample popularity estimation is a drift problem before it is a sampling problem.* A few hundred people give a
> list close to the community's list at that moment; the moment is what is off. The recency-aware sampling and weighting
> we tried repair a small part of that gap (about 2% on the list), so the diagnosis is solid and the repair is open work.
> Personalization keeps improving where popularity has saturated, and choosing the blend inside the sample did not work.

What this framing does not claim: it does not claim breadth beats depth, that the hybrid is safe, that a simple rule
beats in-sample tuning (we never tested one), or that bootstrap bands for rankings are well calibrated (naive ones
are not). The "drift" number compares an oracle of the future with feasible samples, so it is a ceiling on what any
method could recover, not a measured loss.

## 6. Caveats you should read before quoting a number

- **Candidate cap.** Models here choose among 1,500 items only. The share of the panel's real later choices inside
  that universe is 88% on ml-1m but 37% on ml-latest, 11% on amazon-video-games and 6% on gowalla
  (`03_tables/cand_share.csv`). C2 and C3 are read on the registered standard NDCG; the in-universe version is shown
  beside them (`an_c3_checks`).
- **Left-censoring.** Every crossover found at N = 100 in the registered grid is only known to be "at or below 100".
  The extra small-N stage resolves it for some datasets.
- **Panels.** The held-out panel has 194 people on ml-1m and up to 1,500 elsewhere; time ties on ml-1m and Amazon
  reduce time-ordered hiding to partly random hiding.
- **Ties and saturation.** NDCG@100 against a full-community list is close to 1 for large samples, which makes small
  differences hard to read; we report intervals and effect sizes, not only verdicts.
- **Not a real small group.** Samples are random draws from a community, except in the subgroup stage.
- **Reviews.** The code and the confirmatory analysis were reviewed by subagents on the same provider (not an outside
  audit). They found real problems (a nested stale list, biased weights, a cache keyed by object id, a governor that
  could freeze workers) that were fixed and are recorded as pre-registration amendments A1-A15.

- **Panel uncertainty.** Every N* and every C2(b) interval is over 20 draws against one fixed panel; the panel's own sampling
  error (194 people on ml-1m) is not in them. The 132 C2(b) tests are uncorrected.
- **Open follow-ups** suggested by the critics: (1) compare recency methods with the earlier list and the stale list on
  Truth B with the half-life chosen inside the sample; (2) redo C2 and C3 with a second disjoint panel and a bootstrap over
  panel people, and evaluate simple fixed rules ("always personalize at k = all") against the hybrid; (3) lift the
  1,500-item cap on gowalla, amazon-video-games and ml-latest.

## 7. How quiet the run was

Median fan speed 1,463 rpm (idle on this machine is about 1,450), 99th percentile 1,619, maximum 1,742 over
5.3 governed hours; a heavy test before the run pushed the same fans to 2,360 rpm with two workers. The governor froze
the workers twice and never reached the hard limits (smoothed temperature maximum 73.3 C; raw one-second readings reached
81.9 C, 3.4% of samples above 78 C, without the fans reacting, which is why the governor uses a 25-second average). CPU only (no GPU); at most 4 workers (usually 2 or 3); worker memory
peaked at 2.2 GB against an 8 GB budget. The laptop's own swap grew from 26 to 32 GB overnight (other applications) and
free disk dipped to 10.6 GB once (the limit is 10 GB), so check disk space. Figure: `safety_fans_and_temperature.png`; numbers:
`00_safety/safety_summary.json`.
