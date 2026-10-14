# Useful conclusions: Small-Sample Popularity Estimate (negative-result framing)

Chinese version: [`03_conclusions_zh.md`](03_conclusions_zh.md). Full numbers: [`02_results_en.md`](02_results_en.md).
Pre-registered plan: [`01_preregistration_en.md`](01_preregistration_en.md). Tables: `results/small_sample_popularity/03_tables/`; figures: `04_figures/`.

Basis: 11 public datasets, 117,069 small tasks, plus 1,220 fresh-seed replications. Each conclusion has a strength label:
**Confirmed** = a pre-registered test or a replicated, consistent result; **Exploratory** = seen in the data, not pre-registered.

## A. One-sentence position

A popularity list from a few hundred people is close to the whole community's list at that moment. What goes wrong is that
the moment passes. Three simple ideas that sounded right (breadth beats depth, an in-sample hybrid blend, one crossover
size) did not hold as stated, and the reasons are informative.

## B. Conclusions we can argue (ordered by how well supported)

1. **Falsified: an in-sample hybrid blend is not safe (C2). Confirmed.**
   - Within 0.002 of the better pure list in only 70% of 132 cells (falsifier: below 75%).
   - On average worse by 0.0018 NDCG@100 [-0.0029, -0.0008]; better in 1 of 7 datasets.
   - Cause (checked): the in-sample hold-out that picks the blend is right 68% of the time, but always guessing
     "popularity" is already right 67%. It also overstates the gain from personalizing by +0.014 to +0.017.
   - Wording: "choosing the blend inside a small sample has almost no skill beyond the base rate."
2. **Breadth versus depth is a tie in accuracy, but not in repeatability (C1). Confirmed as inconclusive.**
   - Breadth wins 61% of 46 pairs; pooled difference +0.0005 [-0.003, +0.004].
   - Breadth gives a smaller draw-to-draw standard error in 39 of 46 pairs (median ratio 0.88).
   - The unbiased weighted estimator favors depth (-0.008 [-0.012, -0.005]).
   - Wording: "More people buys stability, not accuracy."
3. **Unbiased is not better (X15). Exploratory, consistent.**
   - The statistically unbiased weighted count beats the plain count in only 3 of 50 series, and only at N at least 10%
     of the group.
   - Wording: "In small samples, the textbook-correct estimator loses to the biased one."
4. **A crossover sample size exists, but it is small and not a single number (C3). Mixed.**
   - At k = all it exists in 5 of 7 datasets (needed 5); at k = 10 in 3 (needed 4).
   - Post hoc: at or below 20 people in ml-latest, amazon-software, kuairand-pure; 100 in ml-1m; 1,000 in gowalla;
     none up to 2,000 in the two small Amazon sets.
   - Wording: "N* depends on the dataset more than on history length. No dataset has N* rising with k."
5. **Time is a bigger error than sample size (X10, X7, X12). Exploratory, 10 of 10 datasets.**
   - The gap between the community's earlier list and its later list: 0.66 NDCG@100. Sampling gap at 1,000 people: 0.003.
   - A recent-window sample beats a same-size uniform sample on later choices in 10 of 11 datasets (median +0.037).
   - Recency-weighted counts win 179 of 223 cells (+0.013), but lose 119 of 223 on the old target.
   - Honest limit: the 0.66 is an oracle ceiling, not a measured loss. Recency methods recover about 2% of it.
   - For held-out people the gain is small: median +0.002, negative in gowalla and lastfm-1k.
6. **Popularity saturates, personalization does not (X17). Exploratory.**
   - From N = 100 to the largest N, popularity grows by a factor 1.22, the best personalized model by 1.94.
   - Popularity reaches 95% of its best at least two grid steps earlier in 4 of 7 datasets.
   - Nothing is observed beyond N = 4,000.
7. **Naive rank intervals are overconfident (X5). Exploratory, strong.**
   - Bootstrap interval covers 0.85 of true ranks 51-200 (median); below 0.93 in 79% of cells.
   - Mogstad et al. sets reach 0.95 in every cell but are 3.0 times wider.
   - Wording: "A leaderboard from a small sample needs wider intervals than the bootstrap gives."
8. **A sample cannot judge its own draw (X14). Exploratory.** The bootstrap predicts the level of quality (ratio 1.00 at
   N >= 200) but not which draw is better (Spearman 0.18).
9. **Privacy has a price you can name (X9). Exploratory, mixed.**
   - Central DP at epsilon >= 2 and N >= 2,000 keeps a median 0.86 of the non-private quality.
   - Local DP needs a median 50 times more people (lower bound).
   - Noise removes 96% of top-10 overlap but only 93% of top-100 overlap: not head-concentrated, as predicted.
10. **Subgroups: the prediction failed (X8).** The more a subgroup differs from the global list, the smaller the sample
    at which its own sample wins (Spearman -0.44; -0.20 [-0.37, -0.01] among the 94 series that reached a crossover).
    Part of the larger number is censoring.
11. **Learning speed differs by dataset shape (X3, mixed).** The error exponent correlates with top-heaviness (0.54 at
    k = all, 0.76 at k = 5). A single power law fits only 4 of 11 datasets at k = all.
12. **Surprise hunt.** 40,356 comparisons scanned; the 50 largest eligible re-ran on fresh seeds and all 50 replicated.
    Most are small-N artifacts or breadth-depth interactions; none is claimed as important.

## C. Argument for a negative-result paper

- The three claims were written down and committed before any real run (amendments A1-A15, all dated and listed).
- Two of three failed on pre-registered falsifiers, and we explain why (items 1, 2, 3).
- Side results (items 5, 6, 7) give a usable diagnosis: sampling is not the main error, staleness is.
- Suggested title: *"Small samples are enough for popularity. The list going stale is the problem."*

## D. What not to claim

- Not one N* number; not that breadth beats depth; not that the hybrid is safe.
- Not that recency fixes recover the drift gap (about 2% only).
- Not any of the 50 hunt results as discoveries.
- Not that the shared-provider reviews are an outside audit.

## E. Open work, ranked by value

1. A stop-or-refresh rule: when to resample, evaluated on the later list with the half-life chosen inside the sample.
   This would turn the drift ceiling into a measured loss.
2. Redo C2 and C3 with a second disjoint panel; bootstrap over panel people; test fixed rules such as "always
   personalize at k = all" against the hybrid.
3. Lift the 1,500-item cap on gowalla, amazon-video-games and ml-latest (only 6-37% of later choices fall inside it).
4. X4 and X13 could not be tested on the data we have.

## F. Files in reading order

1. `docs/small_sample/00_start_here_en.md`
2. `docs/small_sample/01_preregistration_en.md`
3. `docs/small_sample/02_results_en.md`
4. this file
5. Figures: `F2-C1_contrasts.png`, `F2-C2a_hybrid_vs_pure.png`, `F2-C3a_crossover_table.png`,
   `an_explore_a_x10_drift_vs_sampling.png`, `an_explore_b_x7_recency.png`, `an_curves_two_speeds.png`
