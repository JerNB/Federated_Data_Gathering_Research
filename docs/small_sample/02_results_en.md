# Results: Small-Sample Popularity Estimate

Chinese version: [`02_results_zh.md`](02_results_zh.md). Pre-registered plan: [`01_preregistration_en.md`](01_preregistration_en.md).
All numbers below come from `results/small_sample_popularity/03_tables/`; every figure is in `04_figures/`.
Run on 11 public datasets, 117,069 small tasks plus 1,220 fresh-seed replications, overnight on one laptop with the fans kept at idle.

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


## Every result, in order

Each block is: the claim, its verdict, a plain reading, the rule that was fixed in advance, and the figures. Numbers come straight from `results/small_sample_popularity/03_tables/`.


### Confirmatory claims


Numbers behind this block: `results/small_sample_popularity/03_tables/an_confirm2_*.csv` and `an_confirm2_verdicts.json`.


#### C1. Breadth beats depth for estimating the list (amended rule A10) — **MIXED**

Literal test (plain count, Truth A, widest cell N at most 5% of the group): breadth beat depth in 28 of 46 judged dataset-budget pairs (61%; rule: at least 75% for support, at most 50% falsifies, in between is inconclusive). Pooled difference (breadth minus depth) +0.0005 NDCG@100 (95% interval -0.0028 to +0.0041; rule: interval above 0 and difference at least 0.005); 6 of 10 datasets have a positive mean difference (most datasets). Secondary lines (same contrast, same rule): ht vs Truth A: 9 of 46 pairs positive (20%), pooled breadth-minus-depth -0.0082 (95% interval -0.0116 to -0.0050) -> not supported | count vs Truth A_u (user-normalised): 29 of 46 pairs positive (63%), pooled breadth-minus-depth +0.0005 (95% interval -0.0031 to +0.0044) -> mixed | boot arm: count vs Truth A: 48 of 106 pairs positive (45%), pooled breadth-minus-depth +0.0008 (95% interval -0.0030 to +0.0044) -> not supported | boot arm: ht vs Truth A: 28 of 106 pairs positive (26%), pooled breadth-minus-depth -0.0114 (95% interval -0.0157 to -0.0073) -> not supported | boot arm: count vs Truth A_u: 49 of 106 pairs positive (46%), pooled breadth-minus-depth +0.0009 (95% interval -0.0029 to +0.0047) -> not supported | boot arm: ht vs Truth A_u: 28 of 106 pairs positive (26%), pooled breadth-minus-depth -0.0115 (95% interval -0.0158 to -0.0074) -> not supported. Mechanism (A9): plain counts at small k estimate the user-normalised list A_u rather than Truth A, so compare the count.A, ht.A and count.Au columns before interpreting the sign.


*Rule fixed in advance:* difference (largest-N cell minus largest-k cell) of count NDCG@100 vs Truth A at equal budget N*k, judged only when the widest cell has N <= 5% of the group: supported if >= 75% of pairs positive AND pooled interval excludes 0 AND pooled difference >= 0.005; falsified if <= 50% positive OR pooled interval entirely below 0; otherwise inconclusive (mixed). Secondary lines: same contrast for ht.A, count.Au and the boot arm (no 5% restriction)


![F2-C1_contrasts.png](../../results/small_sample_popularity/04_figures/F2-C1_contrasts.png)


![F2-C1_count_curves.png](../../results/small_sample_popularity/04_figures/F2-C1_count_curves.png)


#### C2. A hybrid with honest weights is never worse than the better pure list, and sometimes better than both (amended rule A11) — **NOT SUPPORTED**

Over 132 cells, the time-hold-out hybrid was within 0.002 of the better pure list (comparator = the larger cell mean of popularity and personalised_time) in 70% (rule: at least 90%), above that list with a paired 95% interval above 0 in 25% (rule: at least 20%), and the random hold-out put more weight on personalisation than the time hold-out in more draws in 38% of 78 eligible cells (rule: at least 70%; falsified at 50% or below; 12 cells where both hold-outs chose the same model and weight in every draw and 42 cells with fewer than 5 untied draws were excluded). Mean panel NDCG@100 of hybrid_random minus hybrid_time per cell: +0.0003. Share of draws where the hybrid picked w = 0 or 1: 43% (time), 46% (random).


*Rule fixed in advance:* (a) hybrid_time cell mean >= larger of (popularity, personalised_time) cell means - 0.002 in >= 90% of cells; (b) paired-by-draw 95% interval of hybrid_time minus that pure list above 0 in >= 20% of cells; (c) per cell, draws with hybrid_random.w > hybrid_time.w vs <, ties dropped, >= 5 untied draws, identical-split cells excluded; 'random more' larger in >= 70% of eligible cells. Supported if a>=90%, b>=20%, c>=70%; falsified if a<75% or b<10% or c<=50%; else mixed


![F2-C2a_hybrid_vs_pure.png](../../results/small_sample_popularity/04_figures/F2-C2a_hybrid_vs_pure.png)


![F2-C2b_weight_random_vs_time.png](../../results/small_sample_popularity/04_figures/F2-C2b_weight_random_vs_time.png)


#### C3. A crossover sample size exists, and does not grow with k (amended rule A12) — **MIXED**

A crossover size N* (personalised(time) beats popularity, 95% interval above 0 and mean difference above 0 at every larger tested N) exists for k = all in 5 of 7 datasets (rule: at least 5; absent in 2, falsified at 4) and for k = 10 in 3 (rule: at least 4). N*(all) does not exceed N*(10) in any dataset where both are known (no monotonicity violation). Largest N tested per dataset: ml-latest 4000, ml-1m 1000, amazon-video-games 2000, amazon-musical-instruments 2000, amazon-software 4000, kuairand-pure 2000, gowalla 4000. Candidate-universe share below 0.5 (cap-limited): ml-latest, amazon-video-games, amazon-musical-instruments, amazon-software, gowalla.


*Rule fixed in advance:* N*(k), k in {10, all} = smallest N whose paired-by-draw 95% interval of personalised_time - popularity is above 0 AND whose mean difference stays > 0 at every larger tested N; absent counts as larger than any present N*; supported if N* exists at k=all in >= 5 of 7 and at k=10 in >= 4 of 7 and N*(all) > N*(10) never happens; falsified if absent at k=all in >= 4 of 7; else mixed


![F2-C3a_crossover_table.png](../../results/small_sample_popularity/04_figures/F2-C3a_crossover_table.png)


![F2-C3b_personalised_minus_popularity.png](../../results/small_sample_popularity/04_figures/F2-C3b_personalised_minus_popularity.png)


### Companions to C2/C3: in-universe score and smaller sample sizes


Numbers behind this block: `results/small_sample_popularity/03_tables/an_c3_checks_*.csv` and `an_c3_checks_verdicts.json`.


#### C3-in-universe. C3 crossover on the in-universe score (companion to C3, amendment A15) — **MIXED**

Counting only relevant items a model can reach (inside the 1,500-item candidate universe), N* exists at k=all in 6 of 7 datasets (standard score: 5) and at k=10 in 3 (standard: 3). Series whose answer changes: amazon-video-games/k=all. This shows how much of the 'popularity wins' result is the candidate cap talking.


*Rule fixed in advance:* Same N* rule as C3 (A12) on U_ndcg_in; companion, not a registered verdict.


#### C3-small-N. Where does the crossover really start? C3 with the grid extended down to N=20 — **MIXED**

With three smaller sample sizes added (20, 40, 60 people), the crossover at k=all moves below 100 in 3 datasets; ml-latest: <= 20 (smallest tested); ml-1m: 100; amazon-video-games: none (> 2000); amazon-musical-instruments: none (> 2000); amazon-software: <= 20 (smallest tested); kuairand-pure: <= 20 (smallest tested); gowalla: 1000.


*Rule fixed in advance:* N* as in A12, on the grid {20, 40, 60} + the registered grid; exploratory.


![an_c3_checks_small_n.png](../../results/small_sample_popularity/04_figures/an_c3_checks_small_n.png)


### Exploratory: stale list, shrinkage and recency weighting (window truth)


Numbers behind this block: `results/small_sample_popularity/03_tables/an_window_*.csv` and `an_window_verdicts.json`.


#### X1. Exploratory X1 (window truth): the stale list beats a fresh sample from the same window until N is a large share of the group — **NOT SUPPORTED**

Scored on the window (the same stretch of time the fresh sample comes from), a fresh sample with every earlier choice kept (k=all) matched the stale list in 11 of 11 datasets within the sampled N (median 1% of the group where it did). Judged datasets where this took at least 30% of the group: k=all 0/11, k=5 0/9 (top-100 overlap: k=all 0/11, k=5 0/8). Exploratory; few draws per cell make single crossovers noisy.


*Rule fixed in advance:* Crossover N_s >= 30% of the group in most datasets (N_s = smallest N whose mean window-sample count score >= the stale list's score on the window truth; judged on W_ndcg100 at k=5 and k=all, both must agree; W_ov100 reported as a robustness check)


![an_window_x12_decay_minus_count_kall.png](../../results/small_sample_popularity/04_figures/an_window_x12_decay_minus_count_kall.png)


![an_window_x12_win_share_by_dataset.png](../../results/small_sample_popularity/04_figures/an_window_x12_win_share_by_dataset.png)


![an_window_x1_window_stale_vs_fresh.png](../../results/small_sample_popularity/04_figures/an_window_x1_window_stale_vs_fresh.png)


#### X2. Exploratory X2 (window truth): shrinking toward the stale list helps most near the crossover — **NOT SUPPORTED**

Shrinking a fresh window sample toward the stale list gains up to 0.763 NDCG@100 on the window truth, with the gain peaking at a median N of 20. Peak within one grid step of the X1 crossover: k=all 2/11, k=5 1/9 (only datasets whose crossover was reached can be judged). The median shrinkage strength kappa over cells is 16,965; a high kappa means the method leans on the stale list.


*Rule fixed in advance:* Gain of shrinking toward the stale list (shrunk minus count, window score W_ndcg100) peaks near the X1 crossover (peak at the crossover grid point or one grid step away; judged only where the crossover is reached inside the grid; k=5 and k=all must agree, verdict from the majority of judged datasets per k)


![an_window_x2_window_shrinkage_gain.png](../../results/small_sample_popularity/04_figures/an_window_x2_window_shrinkage_gain.png)


#### X12. Exploratory X12: weighting recent choices more helps predict later choices — **SUPPORTED**

Weighting recent choices more beat the plain count list on later real choices (Truth B) in 179 of 223 cells (80%), lost in 2, and was indistinguishable in 42; the median gain over all cells is +0.013 NDCG@100. On the window score it wins 15 and loses 119 of 223, and on the held-out panel it wins 179 and loses 3 of 223. Exploratory, 11 dataset(s) with window samples.


*Rule fixed in advance:* Share of (dataset, N, k) cells where recency-weighted counts beat plain counts on Truth B (B_ndcg100) with the 95% bootstrap interval of the difference above 0 (cells with >= 2 draws): supported if more than 50% of judged cells, not supported if fewer than 25%, else mixed; cells with the interval below 0 are reported as losses


![an_window_x12_decay_minus_count_kall.png](../../results/small_sample_popularity/04_figures/an_window_x12_decay_minus_count_kall.png)


![an_window_x12_win_share_by_dataset.png](../../results/small_sample_popularity/04_figures/an_window_x12_win_share_by_dataset.png)


### Exploratory: scaling, theory check, drift


Numbers behind this block: `results/small_sample_popularity/03_tables/an_explore_a_*.csv` and `an_explore_a_verdicts.json`.


#### X3. Exploratory X3: error falls as a power law with a floor, and the exponent follows the Zipf slope — **MIXED**

Rank correlation between the NDCG@100 error exponent and the Zipf slope is 0.54 at k=all (n=11 datasets) and 0.76 at k=5 (n=11); the prediction was above 0.6. n is small, so these correlations are noisy.


*Rule fixed in advance:* Rank correlation of the power-law exponent and the Zipf slope above 0.6 across the datasets (judged on NDCG@100 error, k=all and k=5; needs >= 5 converged datasets)


![an_explore_a_x3_exponent_vs_zipf.png](../../results/small_sample_popularity/04_figures/an_explore_a_x3_exponent_vs_zipf.png)


![an_explore_a_x3_local_slope.png](../../results/small_sample_popularity/04_figures/an_explore_a_x3_local_slope.png)


#### X4. Exploratory X4: the Dyer-Owen bound is conservative — **NOT TESTABLE**

The observed number of correctly ordered top items is not stored in the e1 rows, so the rule as written cannot be tested. Only a proxy is reported: the predicted resolve-N beside the N where top-10 overlap first reaches 0.8 (k=all). Top-10 overlap reached 0.8 within the sampled N in 10 of 11 datasets, while the bound predicts 15,535 to 21,065,359 people to settle the top-10; proxy outcome: 9 not contradicted, 0 exceeding, 0 uninformative.


*Rule fixed in advance:* Observed correctly ordered depth >= the Dyer-Owen bound at every N (cannot be read from e1 rows; proxy only)


![an_explore_a_x4_dyer_owen_proxy.png](../../results/small_sample_popularity/04_figures/an_explore_a_x4_dyer_owen_proxy.png)


#### X10. Exploratory X10: the drift between Truth A and Truth B is larger than the sampling error — **SUPPORTED**

On the panel score, the drift gap is larger than the sampling gap at N=1000 in 10/10 datasets (median drift 0.021 vs sampling 0.000); on NDCG@100 against the truth lists it is 10/10 (median drift 0.658 vs sampling 0.003). At the largest N the counts are 10/11 and 11/11.


*Rule fixed in advance:* Drift gap (Truth A vs Truth B oracle lists) exceeds the sampling gap in most datasets (judged at N=1000, k=all; measures: panel NDCG@100 and NDCG@100 vs the truth lists)


![an_explore_a_x10_drift_vs_sampling.png](../../results/small_sample_popularity/04_figures/an_explore_a_x10_drift_vs_sampling.png)


### Exploratory: rank sets, biased samplers, recency, subgroups, privacy


Numbers behind this block: `results/small_sample_popularity/03_tables/an_explore_b_*.csv` and `an_explore_b_verdicts.json`.


#### X5 (exploratory): coverage and width of rank intervals, naive bootstrap versus Mogstad sets — **SUPPORTED**

Across 42 cells, the naive intervals cover only 0.85 of true ranks 51-200 (median) and fall below 0.93 in 79% of cells; the Mogstad sets cover 1.00 (median) and reach 0.95 in 100% of cells. The Mogstad sets are 3.0 times as wide as the naive intervals (median).


*Rule fixed in advance:* Naive coverage below 0.93 in the ranks 51-200 band in most cells (> 50% of (dataset,N,k) cells); Mogstad coverage at least 0.95 in most cells.


![an_explore_b_x5_coverage.png](../../results/small_sample_popularity/04_figures/an_explore_b_x5_coverage.png)


![an_explore_b_x5_widths.png](../../results/small_sample_popularity/04_figures/an_explore_b_x5_widths.png)


#### X6 (exploratory): activity-biased samplers, uncorrected count versus weighted (HT) count — **NOT SUPPORTED**

A crossover (uncorrected count ahead or level at the smallest N, weighted count clearly ahead from some N on) was found in: activity: 2 of 11 datasets (10 of 29 dataset-k series); inv_activity: 1 of 11 datasets (4 of 29 dataset-k series). Where it exists, the median crossover is at N = 1000; 40 series never show the weighted count winning.


*Rule fixed in advance:* A crossover N exists in most datasets: the weighted (HT) count beats the uncorrected count (95% CI of the paired difference above 0) from some N on, while the uncorrected count is not behind at the smallest N.


![an_explore_b_x6_activity.png](../../results/small_sample_popularity/04_figures/an_explore_b_x6_activity.png)


![an_explore_b_x6_inv_activity.png](../../results/small_sample_popularity/04_figures/an_explore_b_x6_inv_activity.png)


#### X7 (exploratory): recent-choice samples versus uniform samples on Truth A and Truth B — **SUPPORTED**

recent_rows: Truth A -0.060, Truth B 0.048 (mean difference in NDCG@100, recent minus uniform), sign flip in 8 of 11 datasets; recent_window: Truth A -0.163, Truth B 0.060 (mean difference in NDCG@100, recent minus uniform), sign flip in 10 of 11 datasets. A sign flip means worse on Truth A and better on Truth B.


*Rule fixed in advance:* Opposite signs in most datasets: recent-choice samples are worse than uniform ones on Truth A and better on Truth B (difference of mean NDCG@100, averaged over the (N,k) cells).


![an_explore_b_x7_recency.png](../../results/small_sample_popularity/04_figures/an_explore_b_x7_recency.png)


#### X8 (exploratory): subgroup lists, sample-only versus global versus shrunk, and N** versus distinctiveness — **NOT SUPPORTED**

Across 164 (group, k) series, the Spearman correlation between distinctiveness and N** is -0.44 (95% interval -0.56 to -0.31); the sample-only list never beat the global list in 70 series (counted as above the largest N). The shrunk list was at least as good as the better of sample and global at every N in 7% of series (58% of individual cells).


*Rule fixed in advance:* Positive rank correlation (Spearman, across groups) between distinctiveness (1 - group_vs_global_ov100) and N** (smallest N where the sample-only list beats the global list). Supported if rho > 0 and the 95% bootstrap interval (over groups) excludes 0; mixed if rho > 0 but the interval includes 0.


![an_explore_b_x8_curves.png](../../results/small_sample_popularity/04_figures/an_explore_b_x8_curves.png)


![an_explore_b_x8_nstar_vs_distinctiveness.png](../../results/small_sample_popularity/04_figures/an_explore_b_x8_nstar_vs_distinctiveness.png)


#### X9 (exploratory): privacy noise measured in sample sizes, central versus local, head versus tail — **MIXED**

central DP (eps >= 2, N >= 2000) keeps a median 0.86 of the people's worth; local DP (eps >= 2) needs a median 50 times more people (a lower bound: 87% of these cells are below the nonprivate curve). Privacy noise removes a median 96% of the top-10 overlap but 93% of the top-100 overlap (head loss larger in 49% of cells).


*Rule fixed in advance:* Statements in sample-size ratios, per dataset. Operational reading (fixed in advance): (a) head: private relative loss in ov10 exceeds that in ov100 in most cells; (b) central DP (laplace, gaussian) at eps >= 2 and N >= 2000 has median effective-N ratio >= 0.8; (c) local DP (ldp_oue) at eps >= 2 needs a median of at least 30x more people.


![an_explore_b_x9_effective_n_ratio.png](../../results/small_sample_popularity/04_figures/an_explore_b_x9_effective_n_ratio.png)


![an_explore_b_x9_ndcg_by_mechanism.png](../../results/small_sample_popularity/04_figures/an_explore_b_x9_ndcg_by_mechanism.png)


### Follow-up checks of our first explanations


Numbers behind this block: `results/small_sample_popularity/03_tables/an_followups_*.csv` and `an_followups_verdicts.json`.


#### F1. Breadth makes lists more repeatable even though it does not make them better — **DESCRIPTIVE**

At equal total choices, the draw-to-draw standard error of list quality is smaller for the wide cell in 39 of 46 pairs (median ratio 0.88), while the average quality is the same (C1). Breadth buys stability, not accuracy.


*Rule fixed in advance:* Descriptive follow-up to C1; no registered threshold.


#### F2. Subgroups: the reversed sign survives, but weaker, once never-reached series are dropped — **DESCRIPTIVE**

Among the 94 series where the subgroup sample did beat the global list, the rank correlation between distinctiveness and the sample size at which it did is -0.20 (interval -0.37 to -0.01); the 70 series that never got there are less distinctive on average (0.26 vs 0.39), which is why the first number (-0.44) overstated it.


*Rule fixed in advance:* Descriptive follow-up to X8; no registered threshold.


#### F3. The in-sample hold-out barely beats always guessing the more common outcome — **DESCRIPTIVE**

Personalised beats popularity on the panel in 33% of draws, so always guessing the more common outcome is right 67% of the time; the time-ordered in-sample hold-out is right 68% of the time. It adds little skill.


*Rule fixed in advance:* Descriptive follow-up to X16; no registered threshold.


#### F4. Recency sampling: the community's later list improves a lot, held-out people only a little — **DESCRIPTIVE**

A recent-window sample beats a uniform sample of the same size by a median +0.037 NDCG@100 against the community's later list, but only by +0.0023 for held-out people, and it is negative in 2 of 11 datasets (gowalla, lastfm-1k).


*Rule fixed in advance:* Descriptive follow-up to X7; no registered threshold.


### Exploratory: surprise hunt with fresh-seed replication


Numbers behind this block: `results/small_sample_popularity/03_tables/an_hunt_*.csv` and `an_hunt_verdicts.json`.


#### x11_c1_nonmonotone. Surprise hunt, pattern class 1 non-monotone learning curve (exploratory) — **SUPPORTED**

Of 24159 hypotheses scanned (40356 in all classes), 443 pass the false-discovery control; 20 of the 20 re-run on fresh draws replicated (same sign, one-sided p < 0.05). Only the replicated ones count as findings; all remain exploratory. eligible=256 tested_for_replication=20 


*Rule fixed in advance:* X11 (exploratory, pre-registered): scan every curve for non-monotone shapes and sign reversals; Benjamini-Hochberg q=0.10 within the class; each hit re-run on fresh seeds, reported as found / replicated / not replicated. Class verdict rule (fixed before use): supported = at least half of ALL FDR hits of the class replicate (hits that could not be re-run count as not replicated); mixed = some but fewer; not supported = no FDR hit or none replicated; not testable = hits not yet re-run or cells missing.


#### x11_c2_signflip. Surprise hunt, pattern class 2 sign flip between targets (exploratory) — **NOT TESTABLE**

Of 1872 hypotheses scanned, 253 pass the false-discovery control but none could be re-run because the fresh data lack their cells; none is a finding. eligible=0 tested_for_replication=0 e2 rows hold only panel (U) metrics, so 'hyb_time - pop' cannot be compared with a Truth-A metric: that half of the pre-registered pair is not testable on these data.


*Rule fixed in advance:* X11 (exploratory, pre-registered): scan every curve for non-monotone shapes and sign reversals; Benjamini-Hochberg q=0.10 within the class; each hit re-run on fresh seeds, reported as found / replicated / not replicated. Class verdict rule (fixed before use): supported = at least half of ALL FDR hits of the class replicate (hits that could not be re-run count as not replicated); mixed = some but fewer; not supported = no FDR hit or none replicated; not testable = hits not yet re-run or cells missing.


#### x11_c3_reversal. Surprise hunt, pattern class 3 estimator order reversal (exploratory) — **SUPPORTED**

Of 1630 hypotheses scanned (40356 in all classes), 325 pass the false-discovery control; 6 of the 6 re-run on fresh draws replicated (same sign, one-sided p < 0.05). Only the replicated ones count as findings; all remain exploratory. eligible=6 tested_for_replication=6 


*Rule fixed in advance:* X11 (exploratory, pre-registered): scan every curve for non-monotone shapes and sign reversals; Benjamini-Hochberg q=0.10 within the class; each hit re-run on fresh seeds, reported as found / replicated / not replicated. Class verdict rule (fixed before use): supported = at least half of ALL FDR hits of the class replicate (hits that could not be re-run count as not replicated); mixed = some but fewer; not supported = no FDR hit or none replicated; not testable = hits not yet re-run or cells missing.


#### x11_c4_interaction. Surprise hunt, pattern class 4 interaction (N x k, breadth vs depth) (exploratory) — **SUPPORTED**

Of 7647 hypotheses scanned (40356 in all classes), 1802 pass the false-discovery control; 20 of the 20 re-run on fresh draws replicated (same sign, one-sided p < 0.05). Only the replicated ones count as findings; all remain exploratory. eligible=806 tested_for_replication=20 


*Rule fixed in advance:* X11 (exploratory, pre-registered): scan every curve for non-monotone shapes and sign reversals; Benjamini-Hochberg q=0.10 within the class; each hit re-run on fresh seeds, reported as found / replicated / not replicated. Class verdict rule (fixed before use): supported = at least half of ALL FDR hits of the class replicate (hits that could not be re-run count as not replicated); mixed = some but fewer; not supported = no FDR hit or none replicated; not testable = hits not yet re-run or cells missing.


#### x11_c5_heterogeneity. Surprise hunt, pattern class 5 cross-dataset heterogeneity (exploratory) — **SUPPORTED**

Of 482 hypotheses scanned (40356 in all classes), 189 pass the false-discovery control; 4 of the 4 re-run on fresh draws replicated (same sign, one-sided p < 0.05). Only the replicated ones count as findings; all remain exploratory. eligible=4 tested_for_replication=4 


*Rule fixed in advance:* X11 (exploratory, pre-registered): scan every curve for non-monotone shapes and sign reversals; Benjamini-Hochberg q=0.10 within the class; each hit re-run on fresh seeds, reported as found / replicated / not replicated. Class verdict rule (fixed before use): supported = at least half of ALL FDR hits of the class replicate (hits that could not be re-run count as not replicated); mixed = some but fewer; not supported = no FDR hit or none replicated; not testable = hits not yet re-run or cells missing.


#### x11_c6_valley. Surprise hunt, pattern class 6 mixing valley (exploratory) — **NOT SUPPORTED**

Of 993 hypotheses scanned (40356 in all classes), none survives the false-discovery control, so nothing of this kind is reported as a finding. eligible=0 tested_for_replication=0 


*Rule fixed in advance:* X11 (exploratory, pre-registered): scan every curve for non-monotone shapes and sign reversals; Benjamini-Hochberg q=0.10 within the class; each hit re-run on fresh seeds, reported as found / replicated / not replicated. Class verdict rule (fixed before use): supported = at least half of ALL FDR hits of the class replicate (hits that could not be re-run count as not replicated); mixed = some but fewer; not supported = no FDR hit or none replicated; not testable = hits not yet re-run or cells missing.


#### x11_c7_diverging. Surprise hunt, pattern class 7 diverging targets (exploratory) — **NOT TESTABLE**

Of 2442 hypotheses scanned, 27 pass the false-discovery control but none could be re-run because the fresh data lack their cells; none is a finding. eligible=0 tested_for_replication=0 


*Rule fixed in advance:* X11 (exploratory, pre-registered): scan every curve for non-monotone shapes and sign reversals; Benjamini-Hochberg q=0.10 within the class; each hit re-run on fresh seeds, reported as found / replicated / not replicated. Class verdict rule (fixed before use): supported = at least half of ALL FDR hits of the class replicate (hits that could not be re-run count as not replicated); mixed = some but fewer; not supported = no FDR hit or none replicated; not testable = hits not yet re-run or cells missing.


#### x11_c8_noisehelps. Surprise hunt, pattern class 8 noise helps (exploratory) — **NOT SUPPORTED**

Of 1131 hypotheses scanned (40356 in all classes), none survives the false-discovery control, so nothing of this kind is reported as a finding. eligible=0 tested_for_replication=0 


*Rule fixed in advance:* X11 (exploratory, pre-registered): scan every curve for non-monotone shapes and sign reversals; Benjamini-Hochberg q=0.10 within the class; each hit re-run on fresh seeds, reported as found / replicated / not replicated. Class verdict rule (fixed before use): supported = at least half of ALL FDR hits of the class replicate (hits that could not be re-run count as not replicated); mixed = some but fewer; not supported = no FDR hit or none replicated; not testable = hits not yet re-run or cells missing.


![hunt_best.png](../../results/small_sample_popularity/04_figures/hunt_best.png)


![hunt_counts.png](../../results/small_sample_popularity/04_figures/hunt_counts.png)


### Exploratory: profile length vs training depth, self-assessment, unbiased vs plain count


Numbers behind this block: `results/small_sample_popularity/03_tables/an_extra_*.csv` and `an_extra_verdicts.json`.


#### X13. Exploratory X13: the crossover sample size is set by the served profile length, not by the training depth — **NOT TESTABLE**

Fewer than 4 datasets have a crossover N* for both k_train = 5 and k_train = all at kprof = all (knn100: 3, ease100: 2 of 7 datasets in the data), so the rule cannot be judged yet. The N* values are in the table.


*Rule fixed in advance:* Per dataset and model (knn100, ease100) vs pop at stratum=all, kprof=all: N*(k_train) = smallest N whose paired-by-draw 95% interval of (model - pop) is above 0 and whose mean stays above 0 at every larger N. Supported if N*(k_train=5) is within one grid step of N*(k_train=all) in >= 5 of the datasets that have an N* for both; not supported if they differ by >= 2 steps in >= 4 of them; not testable if fewer than 4 datasets have an N* for both; else mixed. Models are combined (supported only if both are).


![an_extra_x13_nstar_vs_kprof.png](../../results/small_sample_popularity/04_figures/an_extra_x13_nstar_vs_kprof.png)


![an_extra_x13_tercile.png](../../results/small_sample_popularity/04_figures/an_extra_x13_tercile.png)


#### X13b. Exploratory X13b: people with a long history benefit from personalisation at smaller samples than people with a short one — **MIXED**

Heavy-history people reach a crossover at no more than half the sample size of light-history people in: knn100: 1 of 5 judged datasets; ease100: 4 of 5 judged datasets. Where both terciles have an N*, the median heavy/light ratio is 1.00 (9 ratios).


*Rule fixed in advance:* Per dataset and model, kprof=all, k_train=all: ratio = N*(heavy tercile) / N*(light tercile). 'Far smaller' = ratio <= 0.5; a light tercile with no N* counts as N* above the largest tested N (so a heavy N* at most half the largest N counts as far smaller; a heavy N* above that is not judged). Datasets with no N* in either tercile are not judged. Supported if far smaller in strictly more than half of the judged datasets; models combined.


#### X13c. Exploratory X13c: serving each history tercile its own best model beats the best single model — **NOT SUPPORTED**

At the largest N, the tercile policy beats the best single model with a 95% interval above 0 and a positive split-half gain in 0 of 7 datasets (median in-sample gain 0.0000 NDCG@100, split-half 0.0000). The choice is made on the same draws it is scored on, so the in-sample gain is optimistic; the split-half column is the fair check.


*Rule fixed in advance:* Per dataset at the largest N with >= 4 draws, kprof=all, k_train=all: tercile policy = for each tercile the model (of pop, ease100, knn100, ease50, knn50) with the best cell mean; best single policy = the one model with the best mean over the three terciles. Gain = per-draw paired difference of the tercile-average scores. A dataset counts when the 95% interval of the gain is above 0 AND the split-half gain (choose on even draws, score on odd draws, and the reverse) is above 0. Supported if so in strictly more than half of the datasets.


#### X14. Exploratory X14: one sample's own bootstrap predicts its list quality at the right scale, and the row-level bootstrap does not — **MIXED**

The user-level bootstrap has a median pred/real ratio inside [0.67, 1.5] in every N >= 200 cell in 6/6 datasets (median ratio 1.00); the row-level one differs from it by a paired interval that excludes zero in most cells in 6/6 datasets (median ratio 1.00). Across draws, predicted and real quality are rank-correlated at 0.5 or more in 0/6 datasets (median Spearman 0.18). Signed error (pred - real) by N is in the table.


*Rule fixed in advance:* Against Truth A (real_A), cells with N >= 200. (1) user-level bootstrap 'calibrated' in a dataset if the median pred/real ratio over draws lies in [0.67, 1.5] in every (N >= 200, k) cell; supported if so in strictly more than half of the datasets. (2) row-level bootstrap 'mis-scaled relative to the user-level one' in a dataset if the paired-by-draw 95% interval of (pred_row - pred_user) excludes 0 in more than half of its N >= 200 cells; supported if so in strictly more than half of the datasets. (3) Spearman correlation across draws of pred_user vs real_A: a dataset passes if the median over its N >= 200 cells is >= 0.5 (cells with >= 5 draws and non-constant values); supported if so in strictly more than half of the datasets. The three parts are combined (supported only if all are).


![an_extra_x14_self_assessment.png](../../results/small_sample_popularity/04_figures/an_extra_x14_self_assessment.png)


#### X15. Exploratory X15: the unbiased weighted count only beats the plain count at very large samples, if ever — **SUPPORTED**

In 47 of 50 dataset-k series the weighted `ht` count never beats the plain `count` on Truth A, or only does so once N is at least 10% of the group (never: 46; wins earlier than 10% of the group: 3; of the never cases, 0 stop below 10% of the group, so they are lower bounds). At the largest N the median difference (ht - count) is -0.1248 NDCG@100, and ht is ahead in 4 of 50 series.


*Rule fixed in advance:* For each dataset and numeric k: N_win is the smallest N where the independent-draw bootstrap 95% interval of (ht - count) for A_ndcg100 is above 0 (2000 resamples); 'never' if no N qualifies. Supported if, in strictly more than half of the dataset-k series, ht never wins or wins only at N >= 10% of the group. A_u results are shown beside it and do not enter the verdict.


![an_extra_x15_ht_vs_count.png](../../results/small_sample_popularity/04_figures/an_extra_x15_ht_vs_count.png)


### Exploratory: two speeds of learning


Numbers behind this block: `results/small_sample_popularity/03_tables/an_curves_*.csv` and `an_curves_verdicts.json`.


#### X17. Two speeds: the popularity list saturates early, a personalised model keeps improving (exploratory) — **SUPPORTED**

From N=100 to the largest tested N, the popularity list's panel score grows by a median factor of 1.22 while the best personalised model's grows by 1.94; the popularity list reached 95% of its best at least two grid steps earlier in 4 of 7 datasets.


*Rule fixed in advance:* In more than half of the judged datasets, the sample size where the popularity list reaches 95% of its own best is at least two grid steps below that of the best pure personalised model (k = all, panel NDCG@100).


![an_curves_two_speeds.png](../../results/small_sample_popularity/04_figures/an_curves_two_speeds.png)


### Exploratory: a stop rule from inside the sample


Numbers behind this block: `results/small_sample_popularity/03_tables/an_stoprule_*.csv` and `an_stoprule_verdicts.json`.


#### X16. Can the sample tell when personalising starts to pay? (exploratory) — **NOT SUPPORTED**

Hiding each sampled person's latest choices predicts whether personalising helps on the real panel with accuracy 68% (random hiding: 68%). Random hiding overstates the benefit of personalising by +0.017 NDCG on average, time-ordered hiding by +0.014.


*Rule fixed in advance:* Time-ordered in-sample gap predicts the sign of the panel gap with accuracy >= 0.80, and its AUC beats the random hold-out's (interval over datasets above 0).


![an_stoprule_gap_scatter.png](../../results/small_sample_popularity/04_figures/an_stoprule_gap_scatter.png)
