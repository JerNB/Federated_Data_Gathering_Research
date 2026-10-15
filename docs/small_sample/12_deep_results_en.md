# 12. Deep programme: results

## 1. What was run

All 30 pre-registered questions (plan: document 10; one card per question: document 11) ran overnight on 30 logs (13 discovery, 17 holdout; 11 of the 30 are logs that no earlier study touched; 20 are Amazon). 651 result rows, every planned cell present, 0 errors. Machine time: about 2 h 10 min, CPU only, 1 to 3 workers (the governor paused all workers in about 7% of its samples and froze them 10 times). Fans were about 1,460 rpm at the median; the peak was 1,765 rpm, and the highest smoothed temperature in the run was 73.4 C (73.6 C occurred in an earlier, aborted start of the run), slightly over the soft limits (1,750 rpm, 72 C) and under the hard limits (2,100 rpm, 78 C). The raw sensor reached 87.4 C in 14 samples; the limits apply to the smoothed value. Measures: NDCG@10/30/100/300, recall@100, list churn, published list length, split-half agreement, the server's own score.

Score of the registered predictions: paired questions D01 to D14, prediction met in 10 of 14; calculator questions D15 to D30, met in 5 of 16 (D17, D20, D24, D25, D28), not met in 10, not testable in 1 (D21). The predictions that failed are the interesting part.

## 2. Findings that change what you would build

1. **None of the alternatives to the half-life-2 decayed sum gains.** Damped trend, spike clipping, reciprocal-rank, faster forgetting for new items, finer half-life (3 instead of 2), a change detector, an ensemble of half-lives, reserved slots for rising items: all within 0.01 of the baseline or worse. Worst: faster forgetting for young items -0.060, reciprocal-rank -0.022, spike clipping -0.016 (inconclusive), ensemble -0.010. Exceptions to "nothing helps": weighting by round size when sizes vary (finding 2) and the seasonal term in one corner (section 4). A one-log smoke test had shown a reciprocal-rank gain of +0.043; a design review found it came from breaking ties by item id (which follows item age); with random ties it is -0.022 on 26 logs. D03 was reset to "no gain" before the real run.
2. **If the number of reporters changes from round to round, weight each round by its number of reporters.** +0.046 (95% interval +0.037 to +0.054), better in 26 of 26 logs, confirmed on discovery, holdout and the clean replication set (+0.042). Caveat: this fixes a self-made problem. Equal weights are a bad choice when round sizes vary (the sizes here vary by a factor of up to 16, drawn log-normally), so the size of the gain depends on how much sizes vary.
3. **A server can watch its own quality without any ground truth.** Score the published list against the next round's own reporters; across logs this tracks the true quality closely (rank correlation 0.95, median over n values, 245 cells). The self-score is about 0.41 times the true score, so use it for comparing logs and settings at the same n, not as an absolute number. Not tested: whether it follows changes over time within one log.
4. **Privacy was not tested below 1,000 people per round; at 1,000 it already costs a lot, at 10,000 it is partly affordable.** Central privacy (noise once per round, each person limited to 10 rows; epsilon is per round, no composition over rounds): loss against the non-private list is 0.29, 0.14, 0.054, 0.022 NDCG at epsilon 1, 2, 4, 8 with 10,000 people (9 logs); with 1,000 people 0.45, 0.39, 0.31, 0.20. Limiting each person to 3 rows is better at epsilon 8 (0.013 at 10,000, 0.127 at 1,000) because noise grows with the cap. Almost all of the loss is the noise (noise-only loss within 0.003 of the total). Local privacy is not usable here: at epsilon 4 the loss is 0.49 at every n from 1,000 to 10,000, and at epsilon 8 0.28 / 0.23 / 0.12 at n = 1,000 / 3,000 / 10,000.
5. **How old is a 100-person list? It depends on n.** Median equivalent age (a full count that is this many rounds old is as good): 5.2 rounds at n = 30, 3.3 at n = 100, 1.6 at n = 300, 1.1 at n = 1,000. So 1,000 people per round is about as good as a full count one round old. The registered prediction (3 to 6 rounds at n = 100 on the fresh and new logs) was not met: 2.0 on those 7 logs (3.3 on all 26).
6. **Memory should be shorter for short lists.** For a top-10 list the best half-life is no longer than for a top-300 list in 26 of 26 logs, and strictly shorter in 42%.
7. **A published-item threshold destroys a 100-person server.** Requiring 5 decayed reports per item publishes a median of 0.5 items at n = 100 (5.6 at n = 300, 48 at n = 1,000) and only about 22% of them are truly in the top 100; a full 100-item list at threshold 5 is not reached on the grid under the registered rule (n = 3,000 has only 4 logs, below the minimum of 6; in those 4 logs it is 100). With threshold 2 at n = 300, 70 items.

## 3. Findings about whom to ask

- **Asking heavy users more often (probability proportional to the person's rows this round) looks like a gain (+0.036, confirmed on all three registered sets), but it is not a usable idea.** It is an oracle (the server cannot know this before asking), it collects 2.7 times more rows per round on average (median 2.1 times), the gain disappears at n = 300 (+0.004) and reverses at n = 1,000 (-0.019), and the feasible version (probability from last round's rows) loses 0.013 at n = 100 and 0.019 at n = 300. The prediction text in the registry said "about 1.8 times more rows" and quoted the smoke result as +0.067, while the plan document quoted +0.048 for the same smoke run; the measured ratio is 2.7 on average.
- **A rotating panel (keep 75% of last round's active reporters) has no effect** (-0.003, interval -0.007 to +0.001) against a fully fresh panel; the more people are kept the worse (0.1 rotation: -0.007).
- **Samples do not improve a periodic census at 100 people:** adding the samples to a full count taken every 4 rounds loses 0.044 on average (Amazon -0.063, other logs +0.010).

## 4. Findings about time and calendar

- **Same season last year helps only in one corner.** At 90 rounds (about 3 months each) and n = 100 it adds +0.014 (Amazon +0.018, other +0.005; weight 0.25 does slightly better: +0.017); it does nothing at 60 and 45 rounds and loses at n = 300 (-0.008). It did not pass the registered false-discovery filter on the discovery logs, so it is not confirmed.
- **The best half-life in days is not flat.** Only 73% of Amazon cells are within 0.01 of the best for 534 days, 63% for 400, 44% for 700, 35% for 300 days. The registered claim (every value from 300 to 700 days within 0.01 in 80% of cells) was not met, so "use 534 days" is a good default but not a safe one. On the clean set (19 cells from fresh and new logs) the shares are 32%, 63%, 58%, 58% for 300, 400, 534, 700 days.
- **The calendar rule holds on the other logs.** On the 8 non-Amazon logs with enough data the best half-life in days stays within a factor 2 across 45, 60 and 90 rounds (median ratio 1.25).
- **Self-tuning the half-life is a bad idea on long rounds and a fair idea on short rounds.** On Amazon logs at 30 rounds the self-tuning server is worse than half-life 2 by more than 0.01 in 12 of 19 cells; at 90 rounds only 3 of 20. On the other logs it recovers at least half of the available gain in 13 of 23 cases (57%, bar 60%) and is harmful in 2 of 40. Not met by the registered rule.
- **Warm-up (inconclusive):** the registered median of 6 rounds (18 logs) is dominated by logs whose ground truth only becomes usable after round 5, so it measures when evaluation starts, not tracker warm-up. On the 10 logs that can be evaluated from the start the median is 0 (inside the prediction of 3 or less); 8 logs started above 90% and are left out and their true warm-up is unknown. The prediction is reported as not met by the registered rule, but this is not evidence that warm-up is slow.
- **Bad rounds:** the 10th-percentile round is 0.09 NDCG below the mean round at n = 30 and 0.12 at n = 300 (predicted the other way round). The gap does not shrink with n; our interpretation, not tested, is that it comes from drift between rounds rather than sampling noise.

## 5. What the list-churn and agreement meters say

- **List churn between rounds does track staleness across settings** (rank correlation 0.72 over 124 (log, round-length) cells; we expected under 0.5). Caution: this mixes different logs and round lengths, where longer rounds change both churn and staleness, so it does not show that churn is a good alarm inside one log (the correlation with the stability of the best half-life across logs is -0.22).
- **Split-half agreement (overlap of the lists from two random halves of the reporters):** correlation with true quality 0.70 at n = 300 and 0.76 at n = 30 (needed under 0.5 at n = 30), so it is a usable quality meter down to n = 30 too, not only at large n (not at n = 10: 0.34).
- **Conclusions mostly do not depend on the metric:** correlations with the NDCG@100 ordering of the 23 comparable rules are 0.97 (NDCG@300), 0.97 (recall@100), 0.90 (NDCG@30), 0.74 (NDCG@10); one rule changes sign. For top-10 lists, repeat the check.

## 6. Questions that could not be answered

- **People needed for 95% of the ceiling (D21):** only 4 logs have n = 3,000 in the 30-round setting, fewer than the 6 required.
- **Central privacy at n = 10,000 rests on 9 logs;** local privacy likewise.
- **The census break-even (D15):** the sampled list reaches a census taken every 4 rounds above n = 300 (or never within n = 3,000) in 54% of the 24 logs (needed 70%); the median is above 3,000. For a census every 8 rounds, 8% and median 300.

## 7. Limits and disclosures

- **Reviews.** A design critique before launch led to many changes (random tie-breaks, noise only on items already seen, a finer-grid replacement for an arm that was algebraically a longer half-life, feasible variants of the oracle arms, censoring rules, within-n correlations). A code review after the launch started found 3 defects (tie handling at the list cut-off, noise for epsilon 1,000,000, a crash on an empty round); the run was restarted after the fixes. These are same-provider subagents, not an outside audit.
- **Registered predictions set after seeing a one-log smoke test:** D03 (later reset), D12, D13, D15. Their labels are marked in the registry.
- **Exploratory rows.** Secondary arms and non-primary (people, round length) combinations carry no multiplicity control.
- **Amazon dominates:** 20 of the 30 logs. Where the claim is about other logs it is stated.
- **D26 and D22 are thin:** D26's median uses only the n values with at least 12 logs (n = 300, 1,000 and 3,000: 24, 11 and 4 logs), so only n = 300 counts. D22's h = infinity row is not a steady state and decides nothing.
- **Round lengths:** Amazon rounds are 3 to 9 months here; weekly rounds were not tested.

## 8. Files

`configs/deep_registry.json` (frozen before the run), `scripts/ssp/deep_core.py`, `run_deep.py`, `an_deep*.py`, `run_deep_night.sh`; tables `results/small_sample_popularity/03_tables/deep/` (`deep_paired.csv`, `deep_verdicts.json`, one table per question); cards in document 11; raw rows `02_runs/deep.jsonl` (gitignored). Rerun: `nice -n 10 caffeinate -i sh scripts/ssp/run_deep_night.sh`.
