# 11. Deep programme: hypothesis cards (generated)

One card per hypothesis, produced by `scripts/ssp/an_deep.py` from the frozen registry and the run. Tables: `results/small_sample_popularity/03_tables/deep/`.

**Completeness:** every planned cell of every stage is present.

## Scoreboard

| # | Hypothesis | Result |
| --- | --- | --- |
| D01 | Damped trend | NO DIFFERENCE; prediction not met |
| D02 | Clip one-round spikes | INCONCLUSIVE; prediction not met |
| D03 | Reciprocal-rank aggregation | LOSS; prediction met |
| D04 | Fast forgetting for young items | LOSS; prediction met |
| D05 | Finer half-life grid | SMALL LOSS; prediction met |
| D06 | Forget faster when the list jumps | NO DIFFERENCE; prediction met |
| D07 | Ensemble of half-lives | LOSS; prediction not met |
| D08 | Keep the list stable | SMALL GAIN; prediction not met |
| D09 | Reserve slots for rising items | NO DIFFERENCE; prediction met |
| D10 | Same season last year | GAIN; prediction met |
| D11 | Rotating panel | NO DIFFERENCE; prediction met |
| D12 | Ask people in proportion to activity | GAIN; prediction met; CONFIRMED |
| D13 | Weight rounds by their size | GAIN; prediction met; CONFIRMED |
| D14 | Do samples add to a periodic census | LOSS; prediction met |
| D15 | Break-even n against a census | NOT MET |
| D16 | Central DP price list | NOT MET |
| D17 | Local DP price list | MET |
| D18 | How long a list can be published | NOT MET |
| D19 | Equivalent age (replication) | NOT MET |
| D20 | Best half-life by list size | MET |
| D21 | People needed for 95% of the ceiling | NOT TESTABLE |
| D22 | Burn-in | NOT MET |
| D23 | Bad rounds | NOT MET |
| D24 | Does the server's own score track true quality | MET |
| D25 | Does list churn say how stale the list is | MET |
| D26 | Split-half agreement as a quality meter | NOT MET |
| D27 | How flat is the best half-life in days | NOT MET |
| D28 | Is the calendar half-life stable on non-Amazon logs | MET |
| D29 | Self-tuning half-life on other kinds of logs | NOT MET |
| D30 | Do conclusions depend on the metric | NOT MET |

## Cards

### D01. Damped trend (A combine rounds)

- **Question.** Holt damped-trend smoothing (level plus damped trend of the shares) gives a better list than the decayed sum.
- **Registered prediction.** No gain: |mean diff| <= 0.005 for the damped primary at n=100; undamped (p=1.0) loses at least 0.02.
- **Literature anchor.** Gardner 2006: damped trend hard to beat on average, but a trend term is a smoothed difference of noisy levels (variance); local drift slope median 0.72 and H2 (rising items) lost 0.039
- **Result.** NO DIFFERENCE; prediction not met

Primary rule `holt:a=0.3,b=0.1,p=0.9` against `decay:h=2` (NDCG@100, higher is better): average change -0.0022 over 26 logs (95% interval -0.0071 to +0.0027); better in 62% of logs. Amazon logs +0.0023, other logs -0.0145. Plain label: NO DIFFERENCE. Pre-registered test: not confirmed (discovery -0.0030, did not pass the false-discovery filter; holdout -0.0017). Prediction check: damped -0.0022 (upper bound +0.0027); undamped -0.00328; rows per round x1.00 and people x1.00 of the reference. Same direction at 4 of 7 other (people, round length) combinations.

### D02. Clip one-round spikes (A combine rounds)

- **Question.** Clipping each item's count in a round at 3 rows before sharing gives a better list.
- **Registered prediction.** No difference: |mean diff| < 0.003.
- **Literature anchor.** Gardner 2006 s4.1 (trimmed smoothing helped on daily supermarket series); our caps of 30 rows lost 0.0006, so little heavy-tail damage is left
- **Result.** INCONCLUSIVE; prediction not met

Primary rule `clip:c=3` against `decay:h=2` (NDCG@100, higher is better): average change -0.0161 over 26 logs (95% interval -0.0353 to +0.0031); better in 35% of logs. Amazon logs -0.0008, other logs -0.0575. Plain label: INCONCLUSIVE. Pre-registered test: not confirmed (discovery -0.0212, did not pass the false-discovery filter; holdout -0.0123). Prediction check: mean diff -0.0161, 95% interval -0.0353 to +0.0031 (needs inside +/-0.003); rows per round x1.00 and people x1.00 of the reference. Same direction at 7 of 7 other (people, round length) combinations.

### D03. Reciprocal-rank aggregation (A combine rounds)

- **Question.** Adding per-round reciprocal-rank scores 1/(60+rank), decayed with half-life 2, gives a better list than adding shares.
- **Registered prediction.** No gain: the 95% interval of the mean diff excludes +0.01. (A one-log smoke test first showed +0.043; the design review found that it came from breaking ties by item id, which follows item age; with random tie-breaks the same smoke shows -0.026, so the prediction was reset to no gain before the real run.)
- **Literature anchor.** Cormack et al. 2009 RRF k=60 beat score fusion by 4-5% MAP in retrieval; Renda and Straccia 2003 found score sums beat Borda in metasearch
- **Result.** LOSS; prediction met

Primary rule `rrf:k=60` against `decay:h=2` (NDCG@100, higher is better): average change -0.0218 over 26 logs (95% interval -0.0285 to -0.0150); better in 8% of logs. Amazon logs -0.0212, other logs -0.0232. Plain label: LOSS. Pre-registered test: not confirmed (discovery -0.0209, did not pass the false-discovery filter; holdout -0.0221). Prediction check: mean -0.0218, upper 95% bound -0.0150 (a gain of 0.01 is ruled out); rows per round x1.00 and people x1.00 of the reference. Same direction at 7 of 7 other (people, round length) combinations.

### D04. Fast forgetting for young items (A combine rounds)

- **Question.** Items first seen less than 4 rounds ago use half-life 1, older items half-life 4; this beats one half-life for everything.
- **Registered prediction.** No gain (likely a loss): mean diff <= +0.005.
- **Literature anchor.** [INFERENCE] earlier tournament killed every re-weighting of the baseline; item-age mixing splices two score scales
- **Result.** LOSS; prediction met

Primary rule `twospeed:hf=1,hs=4,age=4` against `decay:h=2` (NDCG@100, higher is better): average change -0.0595 over 26 logs (95% interval -0.0782 to -0.0408); better in 8% of logs. Amazon logs -0.0629, other logs -0.0503. Plain label: LOSS. Pre-registered test: not confirmed (discovery -0.0635, did not pass the false-discovery filter; holdout -0.0566). Prediction check: mean -0.0595 (needs <= +0.005), upper 95% bound -0.0408 (a gain of 0.01 is ruled out); rows per round x1.00 and people x1.00 of the reference. Same direction at 7 of 7 other (people, round length) combinations.

### D05. Finer half-life grid (A combine rounds)

- **Question.** A half-life of 3 rounds beats the baseline half-life of 2 rounds (the earlier grid steps by a factor of 2, so the true optimum may lie between grid points).
- **Registered prediction.** No gain: the 95% interval of the mean diff excludes +0.01 (the optimum is flat around 2 rounds). Replaces an empirical-Bayes arm that the design review showed is algebraically a longer half-life.
- **Literature anchor.** Exponential smoothing practice: a flat optimum; earlier tracking study found best half-life 2 in 13 of 14 Amazon logs at the factor-2 grid
- **Result.** SMALL LOSS; prediction met

Primary rule `decay:h=3` against `decay:h=2` (NDCG@100, higher is better): average change -0.0087 over 26 logs (95% interval -0.0174 to -0.0000); better in 31% of logs. Amazon logs -0.0138, other logs +0.0051. Plain label: SMALL LOSS. Pre-registered test: not confirmed (discovery -0.0108, did not pass the false-discovery filter; holdout -0.0072). Prediction check: mean -0.0087, upper 95% bound -0.0000 (a gain of 0.01 is ruled out); rows per round x1.00 and people x1.00 of the reference. Same direction at 5 of 7 other (people, round length) combinations.

### D06. Forget faster when the list jumps (A combine rounds)

- **Question.** When a round's shares differ from the tracker by more than 1.5 times the usual change, halving the tracker's memory gives a better list.
- **Registered prediction.** No gain: mean diff <= +0.005 (the detector rarely fires under slow drift).
- **Literature anchor.** ADWIN (Bifet and Gavalda 2007); Gama et al. 2014 survey: detectors pay off under abrupt change, our measured drift slope is 0.2-1.26 (gradual)
- **Result.** NO DIFFERENCE; prediction met

Primary rule `drift:z=1.5,f=0.5` against `decay:h=2` (NDCG@100, higher is better): average change +0.0008 over 26 logs (95% interval -0.0001 to +0.0018); better in 12% of logs. Amazon logs +0.0007, other logs +0.0010. Plain label: NO DIFFERENCE. Pre-registered test: not confirmed (discovery +0.0000, did not pass the false-discovery filter; holdout +0.0014). Prediction check: mean +0.0008 (needs <= +0.005), upper 95% bound +0.0018 (a gain of 0.01 is ruled out); rows per round x1.00 and people x1.00 of the reference. Same direction at 7 of 7 other (people, round length) combinations.

### D07. Ensemble of half-lives (A combine rounds)

- **Question.** Adding the normalised decayed lists with half-lives 1, 2 and 4 gives a better list than half-life 2 alone.
- **Registered prediction.** No gain on Amazon logs (mean diff <= +0.005); a gain of at least +0.005 on the non-Amazon logs where the best half-life is not 2.
- **Literature anchor.** Brown 2008 / forecast-combination literature: combining smoothers helps mainly when the best parameter is unknown
- **Result.** LOSS; prediction not met

Primary rule `ens` against `decay:h=2` (NDCG@100, higher is better): average change -0.0103 over 26 logs (95% interval -0.0142 to -0.0065); better in 12% of logs. Amazon logs -0.0112, other logs -0.0080. Plain label: LOSS. Pre-registered test: not confirmed (discovery -0.0102, did not pass the false-discovery filter; holdout -0.0105). Prediction check: Amazon -0.0112; other -0.0080; rows per round x1.00 and people x1.00 of the reference. Same direction at 7 of 7 other (people, round length) combinations.

### D08. Keep the list stable (A combine rounds)

- **Question.** Giving items on the previous list a 10% score bonus cuts list churn by at least 30% at a cost of less than 0.005 NDCG.
- **Registered prediction.** Churn falls by >= 30% (relative) and NDCG diff >= -0.005.
- **Literature anchor.** Hysteresis / stability regularisation in top-k maintenance; exact null on NDCG alone is expected
- **Result.** SMALL GAIN; prediction not met

Primary rule `hyst:d=0.1` against `decay:h=2` (NDCG@100, higher is better): average change +0.0024 over 26 logs (95% interval +0.0009 to +0.0038); better in 69% of logs. Amazon logs +0.0022, other logs +0.0028. Plain label: SMALL GAIN.  Prediction check: discovery: churn cut 0.142, NDCG lower bound -0.0006; holdout: churn cut 0.109, NDCG lower bound +0.0007 (needs cut >= 0.30 and lower bound >= -0.005 on both); rows per round x1.00 and people x1.00 of the reference. Same direction at 4 of 7 other (people, round length) combinations.

### D09. Reserve slots for rising items (A combine rounds)

- **Question.** Replacing the last 10 list slots by the items whose recent share (half-life 1) most exceeds their older share (half-life 4) gives a better list.
- **Registered prediction.** No gain: mean diff <= +0.005 (reserved slots cost ranks 91-100 more than rising items earn).
- **Literature anchor.** Bandit-style exploration slots; tournament round 1 (H2) found extra weight for rising items cost 0.039
- **Result.** NO DIFFERENCE; prediction met

Primary rule `reserve:r=10` against `decay:h=2` (NDCG@100, higher is better): average change -0.0009 over 26 logs (95% interval -0.0024 to +0.0006); better in 38% of logs. Amazon logs -0.0004, other logs -0.0022. Plain label: NO DIFFERENCE. Pre-registered test: not confirmed (discovery -0.0000, did not pass the false-discovery filter; holdout -0.0015). Prediction check: mean -0.0009 (needs <= +0.005), upper 95% bound +0.0006 (a gain of 0.01 is ruled out); rows per round x1.00 and people x1.00 of the reference. Same direction at 5 of 7 other (people, round length) combinations.

### D10. Same season last year (A combine rounds)

- **Question.** At 90 rounds (rounds of about 3 months) adding the shares of the same season one year earlier (weight 0.5) gives a better list. Cells where the seasonal term is active in fewer than half of the steady steps are not run.
- **Registered prediction.** Gain at 90 rounds: mean diff >= +0.01 on Amazon logs; no gain on non-Amazon logs. (Earlier year-ago-list test: year-ago list alone was worse in 16 of 16 categories; here it is only an added term.)
- **Literature anchor.** Seasonal naive / Holt-Winters; our S1 test (year-ago list alone, 0 of 16 better); Amazon December peaks
- **Result.** GAIN; prediction met

Primary rule `seas:g=0.5` against `decay:h=2` (NDCG@100, higher is better): average change +0.0142 over 28 logs (95% interval +0.0056 to +0.0228); better in 68% of logs. Amazon logs +0.0179, other logs +0.0048. Plain label: GAIN. Pre-registered test: not confirmed (discovery +0.0110, did not pass the false-discovery filter; holdout +0.0165). Prediction check: Amazon +0.0179 (holdout Amazon +0.0197); other +0.0048; rows per round x1.00 and people x1.00 of the reference. Same direction at 1 of 5 other (people, round length) combinations.

### D11. Rotating panel (B whom to ask)

- **Question.** Keeping 75% of last round's reporters who are active again (rotation rate 0.25; gaps refilled with fresh people) gives a better list than a panel that is refreshed completely (rotation rate 1.0, the same sampler).
- **Registered prediction.** No gain, and a loss that grows with overlap: the 95% interval of the diff excludes +0.01; rho=0.1 worse than rho=0.25 in at least 60% of logs.
- **Literature anchor.** Cantwell 2008, Kish 1999: gain depends on within-unit correlation rho (about 0 for item-level popularity on review logs); plain decayed sum: fresh wins when drift is slow
- **Result.** NO DIFFERENCE; prediction met

Primary rule `decay:h=2@panel:rho=0.25` against `decay:h=2@panel:rho=1.0` (NDCG@100, higher is better): average change -0.0032 over 26 logs (95% interval -0.0074 to +0.0010); better in 46% of logs. Amazon logs -0.0017, other logs -0.0073. Plain label: NO DIFFERENCE. Pre-registered test: not confirmed (discovery -0.0018, did not pass the false-discovery filter; holdout -0.0042). Prediction check: rho=0.25 diff -0.0032 (upper bound +0.0010); rho=0.1 worse than 0.25 in 65% of logs; rows per round x1.08 and people x1.00 of the reference. Same direction at 7 of 7 other (people, round length) combinations.

### D12. Ask people in proportion to activity (B whom to ask)

- **Question.** Drawing the n reporters with probability proportional to their rows in the round (an oracle: the server cannot know this before asking) gives a better list than drawing them uniformly; a feasible variant uses last round's rows.
- **Registered prediction.** Oracle arm: gain of at least +0.01 at n=100, but with about 1.8 times more rows collected per round (a one-log smoke showed +0.067 with 1.8x rows); the feasible arm (last round's rows) is secondary and expected to show no gain. Rows per round are reported next to every difference.
- **Literature anchor.** Hansen-Hurwitz / Horvitz-Thompson: with p proportional to size and an uncorrected pooled count the estimate is size-biased, but the target (row counts) is itself dominated by heavy users; the oracle also collects more rows per person asked (reported as rows per round)
- **Result.** GAIN; prediction met; CONFIRMED

Primary rule `decay:h=2@pps` against `decay:h=2` (NDCG@100, higher is better): average change +0.0361 over 26 logs (95% interval +0.0213 to +0.0509); better in 88% of logs. Amazon logs +0.0476, other logs +0.0049. Plain label: GAIN. Pre-registered test: CONFIRMED (discovery +0.0355, holdout +0.0344, clean replication set +0.0142 over 6 logs). Prediction check: holdout label GAIN (holdout mean +0.0344); all-log label GAIN; rows per round x2.73 and people x1.00 of the reference. Same direction at 5 of 7 other (people, round length) combinations.

### D13. Weight rounds by their size (B whom to ask)

- **Question.** When the number of reporters varies from round to round, weighting each round by its number of reporters beats weighting rounds equally.
- **Registered prediction.** Gain: mean diff >= +0.01 against equal weighting under the same variable round sizes (set after a one-log smoke showed +0.033).
- **Literature anchor.** Inverse-variance weighting (standard); variance of a round's share vector is proportional to 1/n_w
- **Result.** GAIN; prediction met; CONFIRMED

Primary rule `ivw:g=1@varn` against `decay:h=2@varn` (NDCG@100, higher is better): average change +0.0458 over 26 logs (95% interval +0.0373 to +0.0543); better in 100% of logs. Amazon logs +0.0486, other logs +0.0381. Plain label: GAIN. Pre-registered test: CONFIRMED (discovery +0.0422, holdout +0.0467, clean replication set +0.0419 over 6 logs). Prediction check: holdout label GAIN (holdout mean +0.0467); all-log label GAIN; rows per round x1.00 and people x1.00 of the reference. Same direction at 7 of 7 other (people, round length) combinations.

### D14. Do samples add to a periodic census (G census plus samples)

- **Question.** If a full count is available every 4 rounds, adding the continuous n=100 samples (the census counts 8 times one round's sample) to the census beats using the census alone.
- **Registered prediction.** No gain at n=100 (mean diff <= +0.005 against the stale census list); a gain at n=1000 would be reported as exploratory.
- **Literature anchor.** Composite / benchmark estimators need the sample to carry information the benchmark lacks; at n=100 a 4-round-old full count already beats the sample (eq. age 4.4 rounds)
- **Result.** LOSS; prediction met

Primary rule `cens:m=4,w=8` against `censold:m=4` (NDCG@100, higher is better): average change -0.0435 over 26 logs (95% interval -0.0696 to -0.0175); better in 35% of logs. Amazon logs -0.0633, other logs +0.0101. Plain label: LOSS. Pre-registered test: not confirmed (discovery -0.0346, did not pass the false-discovery filter; holdout -0.0501). Prediction check: mean -0.0435 (needs <= +0.005), upper 95% bound -0.0175 (a gain of 0.01 is ruled out); rows per round x1.00 and people x1.00 of the reference. Same direction at 4 of 7 other (people, round length) combinations.

### D15. Break-even n against a census (G census plus samples)

- **Question.** The number of people per round at which the sampled tracker matches a full count taken every 4 or 8 rounds.
- **Registered prediction.** Break-even n (first n on the grid with NDCG(sampled) >= NDCG(censold m=4)) is above 300 in at least 70% of eligible logs.
- **Literature anchor.** [INFERENCE] equivalent age 4.4 rounds at n=100 (tracking study)
- **Result.** NOT MET

The sampled tracker is compared with a full count taken every 4 rounds (the 'census') in 24 logs. The smallest number of people per round at which the sampled list is at least as good is above 300 (or never reached on the grid up to 3000) in 54% of logs; the median break-even is >3000 people (for a census every 8 rounds: 8% above 300, median 300). The prediction needed at least 70% above 300, so it is not met.

### D16. Central DP price list (C privacy)

- **Question.** How many people per round are needed for central differential privacy (each person limited to 10 rows) to cost less than 0.05 NDCG.
- **Registered prediction.** At n=10000 (90 rounds), eps=2 per round loses < 0.05 against the non-private baseline in at least 50% of eligible logs; at n=1000, eps=8 does NOT reach < 0.05 in at least 50% of logs (any round length). Also reported: noise-only loss against the capped, noise-free reference (eps=1e6) on the same people.
- **Literature anchor.** Balle-Wang analytic Gaussian (sigma 3.7/2.0/1.08 at eps 1/2/4 for L2 sensitivity 1); Desfontaines et al. / Korolova: thresholds need about 12 users at eps=1; amplification by subsampling not applied here (conservative)
- **Result.** NOT MET

Central privacy (noise added once to the counts, each person limited to 10 rows) is compared with the same list without noise. With 10000 people per round and privacy level eps=2, the loss is below 0.05 NDCG in 11% of 9 logs; with 1000 people per round and eps=8 the loss is below 0.05 in 17% of 36 logs. The prediction needed at least 50% in the first case and fewer than 50% in the second, so it is not met.

### D17. Local DP price list (C privacy)

- **Question.** Local differential privacy (optimised unary encoding, one row per person) at n=10000 loses more than 0.1 NDCG even at eps=4.
- **Registered prediction.** At n=10000 (90 rounds), eps=4 loses > 0.1 against the non-private baseline in at least 50% of eligible logs. Also reported: loss against the one-row-per-person noise-free reference.
- **Literature anchor.** Wang et al. 2017: Var = 4 n e^eps/(e^eps-1)^2 per count (std 0.19 of share at eps=1, n=100 per LitPrivacy); local DP needs about 50-100x more people in our main study (X9)
- **Result.** MET

Local privacy (each person randomises their own report, so the server never sees true counts) is compared with the same list without noise. With 10000 people per round and privacy level eps=4, the list loses more than 0.1 NDCG in 89% of 9 logs (mean loss 0.493, median 0.582). The prediction needed at least 50% of logs, so it is met.

### D18. How long a list can be published (C privacy)

- **Question.** With a minimum running count of 5 reports per published item, a server with 100 people per round can publish fewer than 50 items.
- **Registered prediction.** At n=100, t=5: median list length (len100 metric) < 50 and precision of the published items (prec100) >= 0.5.
- **Literature anchor.** LitPrivacy: deployed thresholds 10-50 users (Chrome k=10-50, TrieHH theta=10-17); publishable items <= n*c/tau
- **Result.** NOT MET

A published item must have at least 5 reports in the running count. With 100 people per round, the median published list has 0.5 items (the full list would be 100) and 22% of the published items are truly in the top 100 (median over 26 logs). A full 100-item list is first reached at no n on the grid up to 3000. The prediction needed fewer than 50 items and at least 50% correct items, so it is not met.

### D19. Equivalent age (replication) (D planning)

- **Question.** A list from 100 people per round is worth a full count that is 3 to 6 rounds old, on the new categories and the new non-Amazon logs too.
- **Registered prediction.** Median equivalent age at n=100 over fresh and new logs lies in [3, 6] rounds.
- **Literature anchor.** Tracking study (equivalent age 1.3-8, median 4.4 on Amazon)
- **Result.** NOT MET

The 'equivalent age' is how old a full count of all users would have to be to be as good as the sampled list: it is the lag at which the full count's NDCG equals the sampled list's NDCG (interpolated between lags 0 to 8, capped at 0 and 8). With 100 people per round, the median equivalent age over 7 fresh-category and new non-Amazon logs is 1.97 rounds (1 logs capped at 0, 0 capped at 8; median over all logs 3.31). The prediction needed a median between 3 and 6, so it is not met.

### D20. Best half-life by list size (D planning)

- **Question.** The best half-life is shorter for a top-10 list than for a top-300 list.
- **Registered prediction.** In at least 60% of logs the best half-life for k=10 is <= the best half-life for k=300 (ties count as met), and strictly shorter in at least 25%.
- **Literature anchor.** [INFERENCE] head items turn over faster than the tail
- **Result.** MET

For each log we find the half-life (how fast old reports fade) that gives the best list at n=100 people per round, once for a top-10 list and once for a top-300 list. In 100% of 26 logs the best half-life for the top 10 is no longer than for the top 300, and it is strictly shorter in 42%. The prediction needed at least 60% and at least 25%, so it is met.

### D21. People needed for 95% of the ceiling (D planning)

- **Question.** Even 3000 people per round do not reach 95% of what a full count of all rounds with the same half-life achieves, in most logs.
- **Registered prediction.** For k=100, NDCG(baseline, n=3000) < 0.95 x NDCG(fulldecay h=2) in at least 70% of logs eligible at n=3000.
- **Literature anchor.** Dyer-Owen style bounds / main study: drift not sampling dominates; equivalent age 4.4 at n=100
- **Result.** NOT TESTABLE

Not testable: only 4 log(s) eligible at n=3000, at least 6 are needed.

### D22. Burn-in (D planning)

- **Question.** With half-life 2 the list reaches 90% of its steady-state NDCG within 3 rounds at n=100 (a new tracker is useful after 3 rounds).
- **Registered prediction.** Median round at which mean NDCG (over draws) first reaches 90% of the steady mean is <= 3 for h=2 at n=100 (logs whose first scored round is already above 90% are left-censored and excluded). The h=inf row is reported but decides nothing (its list is not a steady state).
- **Literature anchor.** Exponential smoothing: effective memory 1/(1-2^(-1/h)) = 3.4 rounds at h=2
- **Result.** NOT MET

Burn-in is the first round at which the list's average NDCG reaches 90% of its steady level (steady = average over rounds 8 and later), at 100 people per round. With half-life 2 the median round is 6.0 (18 logs); with no forgetting (half-life infinite) it is 0.0 (12 logs); logs whose first evaluated round was already at 90% after round 3 are left out (8 and 14). The prediction needed at most 3 for half-life 2 and at least 4 without forgetting, so it is not met.

### D23. Bad rounds (D planning)

- **Question.** At n=300 the 10th-percentile round is within 0.10 NDCG of the mean round; at n=30 it is more than 0.15 below.
- **Registered prediction.** Median over logs of (mean - 10th percentile of per-round NDCG over steps >= 8 and draws): <= 0.10 at n=300 and >= 0.15 at n=30.
- **Literature anchor.** [INFERENCE] sampling noise shrinks as 1/sqrt(n); round-to-round shifts are part of drift
- **Result.** NOT MET

A 'bad round' is a round whose NDCG is in the worst 10% of rounds (steady rounds 8 and later, all draws pooled); we measure how far it sits below the average round. The median gap over logs is 0.121 NDCG at 300 people per round (24 logs) and 0.091 at 30 people per round (28 logs). The prediction needed at most 0.10 at n=300 and at least 0.15 at n=30, so it is not met.

### D24. Does the server's own score track true quality (E self-monitoring)

- **Question.** NDCG of the published list against the next round's own sample (no ground truth needed) is rank-correlated with the true NDCG at 0.7 or more across logs, round lengths and n.
- **Registered prediction.** MEDIAN over n values (those with at least 12 cells) of the within-n Spearman correlation across (log, n_win) cells between steady mean selfnd and steady mean nd100 of the baseline is >= 0.7; the pooled correlation across n is also reported (it is inflated because both rise with n).
- **Literature anchor.** LitMonitoring: the next-round sample gives an unbiased unnormalised DCG (prequential scheme, Gama et al. 2013); the sample-ideal normaliser is biased upward
- **Result.** MET

Across 245 cells we compared the server's own score of its list (scored against the next round's reporter sample, no ground truth needed) with the true NDCG@100 (0 to 1, higher means closer to the real top 100). The Spearman rank correlation (1 = same ordering, 0 = unrelated) is 0.95 (median of the within-n correlations over 4 values of n; pooled over all n it is 0.94, inflated because both scores rise with n) against a bar of 0.7, so the prediction is MET. The self-score is on average 0.41 times the true score, and this ratio moves with n with rank correlation 0.66.

### D25. Does list churn say how stale the list is (E self-monitoring)

- **Question.** The churn of the published list between rounds is rank-correlated at 0.5 or more with how far the list is behind a full count (NDCG(fulldecay h=2) minus NDCG(baseline)).
- **Registered prediction.** Not met expected: |Spearman| < 0.5 across (log, n_win) cells at n=100 (literature: churn measures noise and inertia, not staleness).
- **Literature anchor.** Webber et al. 2010 (RBO between successive lists); Garg et al. 2022 (accuracy without labels is as hard as learning); decayed sums mechanically lower churn at longer half-lives
- **Result.** MET

Across 124 cells at n=100 we compared how much the published top-100 changes from one round to the next (churn, share of the list replaced) with how far the list is behind a full count (NDCG of the full-count list minus NDCG of the baseline). Their Spearman rank correlation is 0.72; the statement needs a size of 0.5 or more, so it is MET (this study expected NOT MET). Against how much the best half-life in days differs between round counts, the log-level correlation is -0.22 over 29 logs.

### D26. Split-half agreement as a quality meter (E self-monitoring)

- **Question.** The overlap between top-100 lists built from two random halves of the reporters is rank-correlated at 0.7 or more with the true NDCG, for n of 300 or more.
- **Registered prediction.** MEDIAN over n in {300,1000,3000} (those with at least 12 cells) of the within-n Spearman(agree, nd100) across logs is >= 0.7; at n=30 the within-n correlation is below 0.5.
- **Literature anchor.** LitMonitoring: split-half is the noise floor; Mogstad et al. 2022 rank confidence sets; informative only when n >= about 400
- **Result.** NOT MET

Split-half agreement is the overlap of the top-100 lists built from two random halves of the reporters. Its Spearman rank correlation (1 = same ordering) with the true NDCG@100 is 0.70 (median over the 1 value(s) of n in 300, 1000, 3000 that have at least 12 logs) (bar 0.7) and 0.76 at n=30 over 28 cells (must be below 0.5). The prediction is NOT MET.

### D27. How flat is the best half-life in days (F calendar)

- **Question.** On Amazon logs any half-life from 300 to 700 days is within 0.01 NDCG of the best of the days grid, for rounds of 3 to 9 months.
- **Registered prediction.** For at least 80% of Amazon (log, n_win in {30,45,60,90}) cells the NDCG regret of each of 300, 400, 534, 700 days is <= 0.01 relative to the best of the days values. Reported separately for the clean set (holdout_fresh and new logs) because the anchor numbers come from older logs.
- **Literature anchor.** Our tournament rounds 3 (400 days +0.057, 534 days +0.054, 700 days +0.031 at 90 rounds on holdout)
- **Result.** NOT MET

On 79 Amazon cells (log x round length of 3 to 9 months, 20 logs) we measured how much NDCG@100 each half-life in days loses against the best of the eight tried values (the regret). The share of cells with regret of at most 0.01 is 0.35 for 300 days, 0.63 for 400 days, 0.73 for 534 days, 0.44 for 700 days; every one must reach 0.8, so the prediction is NOT MET.

### D28. Is the calendar half-life stable on non-Amazon logs (F calendar)

- **Question.** On each non-Amazon log the best half-life in days stays within a factor of 2 across 45, 60 and 90 rounds.
- **Registered prediction.** At least 60% of eligible non-Amazon logs have max/min best days <= 2 across n_win 45, 60, 90 (decay:days arms only). A log is censored (left out, counted) when its best days is at the edge of the grid (3 or 1500 days) at any of the three widths or when all days arms are within 0.001 NDCG of each other.
- **Literature anchor.** Earlier calendar check (11 of 18 logs same in days; Amazon 494-587; Gowalla 83); [INFERENCE] growing platforms prefer shorter spans
- **Result.** MET

For each of 8 non-Amazon logs we found the best half-life in days at 45, 60 and 90 rounds and divided the largest by the smallest. 8 logs (1.00) stay within a factor of 2 against a bar of 0.6, so the prediction is MET. The median ratio is 1.25 and the largest is 2.00; old logs 1.00, new logs 1.00.

### D29. Self-tuning half-life on other kinds of logs (F calendar)

- **Question.** On non-Amazon logs a server that picks its half-life from its own samples recovers at least half of the gain of the best fixed half-life over half-life 2 in 60% of cases and is never worse than 0.01; on Amazon logs it is worse than 0.01 in at least 30% of cases.
- **Registered prediction.** Non-Amazon (log, n_win) cells at n=100: share with (selfsel - base) >= 0.5 x (best fixed - base) is >= 0.6 among cells where best fixed - base >= 0.01, and no cell has selfsel - base < -0.01; Amazon cells: share with selfsel - base < -0.01 is >= 0.3.
- **Literature anchor.** Tracking study PD (Other stratum 9 of 12 cells recovered half, 0 harmful; Amazon harmful in 51% of 59 cells)
- **Result.** NOT MET

On non-Amazon logs, among 23 cells where the best fixed half-life beats half-life 2 by 0.01 NDCG or more, the self-tuning server recovers at least half of that gain in 13 (0.57; bar 0.6), and it is worse than half-life 2 by more than 0.01 in 2 of 40 non-Amazon cells (must be 0). On Amazon logs it is that much worse in 38 of 84 cells (0.45; bar 0.3). The prediction is NOT MET.

### D30. Do conclusions depend on the metric (H metric choice)

- **Question.** The ordering of the 30-odd aggregation arms by mean difference is the same under recall@100, NDCG@10, NDCG@30 and NDCG@300 as under NDCG@100 (rank correlation 0.9 or more), and no gain or loss verdict flips sign.
- **Registered prediction.** Among agg arms other than the oracle and census arms (fulldecay, cens, censold) and with |mean diff in NDCG@100| <= 0.02, the Spearman between mean diffs (arm minus baseline, n=100, 30 rounds, all logs) under NDCG@100 and under each other metric is >= 0.9; number of arms whose sign flips with |diff| > 0.005 under NDCG@100 is 0.
- **Literature anchor.** LitMonitoring: NDCG@k vs recall@k vs RBO agree on large effects, can disagree on small ones
- **Result.** NOT MET

We ranked 23 aggregation arms by their mean NDCG@100 difference from the baseline over 29 logs at n=100, and repeated this with NDCG@10, NDCG@30, NDCG@300 and recall@100. The Spearman rank correlations (1 = same ordering) with the NDCG@100 ordering are 0.74 (nd10), 0.90 (nd30), 0.97 (nd300), 0.97 (rec100), all required to be at least 0.9. 1 arms flip sign with an NDCG@100 difference above 0.005 (must be 0), so the prediction is NOT MET.
