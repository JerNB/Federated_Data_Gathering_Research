# 14. Can the server choose its own memory? (post-hoc exploration after doc 13)

*Exploratory and not pre-registered: written after the W1–W11 verdicts and after finding 6 of doc 13. Nothing here has been checked on data that were not already used. Script: `scripts/ssp/explore_selfsel_weekly.py`. Tables: `results/small_sample_popularity/03_tables/weekly/i_*.csv`.*

## 0. In one paragraph

At 7-day rounds, a server that keeps several memories (half-lives) in parallel and each round publishes the one whose recent **self-score** is best (its own earlier list scored on the next round's reporters, so no ground truth is needed) loses 0.0027 NDCG against the best fixed memory chosen in hindsight. A table fitted on all other logs loses 0.0291, a table fitted on the other logs of the same group (Amazon or other) loses 0.0161, and "2 rounds" loses 0.1350 (95 cells from 27 logs; the tables are fitted leave-one-log-out). The selector has no fitted parameter. It can fail: at 28-day rounds with 30 people it is worse than the tables, and on the cells of the earlier self-tuning test (D29: n = 100, 20 to 90 rounds per log, Amazon rounds of about 3 to 16 months) it is worse than 2 rounds by more than 0.01 in 37% of Amazon cells (63% at 30 rounds per log). All of this is exploratory.

## 1. What was done (plain words)

1. **The selector.** The server keeps one decayed list per half-life option (the options already in the stored cells: 7 to 896 or 1,792 days at 7-day rounds). When a list has been published, it is scored against the reporters of the next round: its self-score. At step t the server publishes the option with the best weighted average of its self-scores from steps before t. The weights halve every 26 rounds (half a year of weekly rounds); the author chose 26 before the first run and did not tune it (no commit records this; finding 3 shows other values, and a plain average is reported too). Every score used at step t was available when the list for step t was published. The published list is then scored, as in all earlier studies, against the full counts of the next round.
2. **No new simulation.** The stored weekly cells (27 logs, n = 30, 100, 300, 1,000, rounds of 7 and 28 days, 95 cells at each round length; the n = 3,000 cells, 7 logs, are left out) hold the NDCG and the self-score of every option at every step and draw. The rule is recomputed from them.
3. **Regret.** The best fixed option in the same cell (chosen with the truth; mean over steady steps and draws) minus the value of the rule. Smaller is better.
4. **Comparators.** "2 rounds"; a table of rounds fitted on all other logs; a table fitted on the other logs of the same group. The tables are fitted leave-one-log-out and may pick 1 to 256 rounds; an option that does not exist in a cell snaps to the nearest one. Cells at n = 30 and 300 have options up to 1,792 days (256 rounds at 7-day rounds); cells at n = 100 and 1,000 have options up to 896 days (128 rounds).
5. **Replay on the earlier self-tuning test (D29).** Its stored cells (n = 100, 20 to 90 rounds per log; Amazon rounds are about 3 to 16 months long, the other logs' rounds range from 0.3 days to 17 months) also hold the self-score of every option, so the same rule can be replayed on them.
6. **Calculator.** A regression of the best memory on n, reports per person and a drift measure was compared at 7-day rounds, leave-one-log-out.

## 2. Findings

1. **At 7-day rounds the selector is close to the best fixed memory.** Mean regret (95 cells: 70 Amazon, 25 other):

   | Rule | Amazon | Other | All |
   |---|---:|---:|---:|
   | 2 rounds | 0.1656 | 0.0493 | 0.1350 |
   | Pooled table (fitted on all other logs) | 0.0248 | 0.0413 | 0.0291 |
   | Same-group table | 0.0165 | 0.0148 | 0.0161 |
   | Self-score selector (weights halve every 26 rounds) | 0.0031 | 0.0015 | 0.0027 |
   | Self-score selector (plain average) | 0.0039 | 0.0014 | 0.0032 |

   By n (all groups; 27 / 27 / 24 / 17 cells for n = 30 / 100 / 300 / 1,000): selector 0.0057 / 0.0025 / 0.0011 / 0.0003; same-group table 0.0091 / 0.0187 / 0.0189 / 0.0192; pooled table 0.0143 / 0.0405 / 0.0359 / 0.0252; 2 rounds 0.1852 / 0.1722 / 0.1010 / 0.0442. Paired over the 27 logs (each log's cells averaged): selector minus pooled table −0.0274 (95% interval −0.0418 to −0.0154; the selector is better in 93% of logs), minus same-group table −0.0141 (−0.0241 to −0.0059; 70%), minus 2 rounds −0.1291 (−0.1576 to −0.0998; 100%). No cell is worse than 2 rounds by more than 0.01 (largest: 0.0029).

2. **At 28-day rounds the edge is smaller, and at 30 people the tables win.** Mean regret (95 cells):

   | Rule | Amazon | Other | All |
   |---|---:|---:|---:|
   | 2 rounds | 0.1084 | 0.0344 | 0.0889 |
   | Pooled table | 0.0107 | 0.0321 | 0.0164 |
   | Same-group table | 0.0107 | 0.0172 | 0.0124 |
   | Self-score selector (weights halve every 26 rounds) | 0.0094 | 0.0024 | 0.0075 |

   By n: selector 0.0160 / 0.0061 / 0.0033 / 0.0023; same-group table 0.0054 / 0.0112 / 0.0215 / 0.0127; pooled table 0.0060 / 0.0167 / 0.0283 / 0.0154; 2 rounds 0.0948 / 0.1237 / 0.0817 / 0.0345. Paired over logs: selector minus pooled table −0.0099 (−0.0213 to −0.0009; better in 52% of logs), minus same-group table −0.0054 (−0.0119 to +0.0003; 48%), so it is not clearly better than a same-group table. No cell is worse than 2 rounds by more than 0.01 (largest: 0.0088).

3. **Checks at 7-day rounds.**
   - Half-life of the average (6 / 13 / 26 / 52 / 104 rounds, plain average): 0.0035 / 0.0027 / 0.0027 / 0.0028 / 0.0030 / 0.0032. At 28-day rounds: 0.0075 / 0.0074 / 0.0075 / 0.0077 / 0.0078 / 0.0080.
   - Controls on steps 40 and later, regret against the best fixed option on those steps (selector 0.0018, 2 rounds 0.1433): a random option each step 0.0844; the selector driven by another log's self-scores (mean over all other logs) 0.0502; the choice frozen after 8 / 12 / 20 / 40 weeks 0.0582 / 0.0406 / 0.0260 / 0.0110. So the self-scores carry log-specific information, and the selector needs a long history and keeps updating.
   - Warm-up, regret against the cell's best fixed option by step (pooled table in brackets): steps 8–20 0.0067 (0.0112); 20–40 0.0059 (0.0184); 40–80 0.0029 (0.0276); 80 and later 0.0011 (0.0357).
   - Option set: all options (factor 2) 0.0027; factor 4 (14, 56, 224, 896 days) 0.0047; factor 8 (14, 112, 896 days) 0.0073.
   - Second half of the steady steps (regret against the second half's best): selector 0.0016; a fixed option chosen with the truth on the first half 0.0022; pooled table 0.0363; 2 rounds 0.1387.

4. **Replay on the 20 to 90-round cells of D29: on Amazon logs the failure is real.** Difference from "2 rounds" in NDCG@100 at n = 100, and the share of cells worse than 2 rounds by more than 0.01:

   | Rounds per log | Group (cells) | Best fixed half-life in rounds | D29 `selfsel` | Selector (options in rounds and days) | Share worse than 2 rounds: D29 / selector |
   |---:|---|---:|---:|---:|---|
   | 30 | Amazon (19) | +0.0022 | −0.0221 | −0.0192 | 63% / 63% |
   | 45 | Amazon (20) | +0.0074 | −0.0237 | −0.0167 | 65% / 60% |
   | 60 | Amazon (20) | +0.0271 | −0.0139 | −0.0013 | 50% / 25% |
   | 90 | Amazon (20) | +0.0543 | +0.0172 | +0.0357 | 15% / 10% |
   | all | Amazon (84) | +0.0218 | −0.0099 | −0.0002 | 45% / 37% |
   | all | other (40) | +0.0343 | +0.0218 | +0.0268 | 5% / 0% |

   The 20-round cells (5 Amazon, 5 other) show no harm; all rows are in `i_selfsel_d29_replay.csv`. The "other" rows mix very different round lengths (0.3 days to 17 months). With 30 to 45 rounds of months each, 2 rounds is already close to the best fixed option on Amazon logs, and the selector, like D29's rule, picks worse options in most cells. With more rounds (60, 90) the harm shrinks. On the other logs it recovers most of the gain of the best fixed option.

5. **The two-feature calculator is not better than a domain label.** Mean regret at 7-day rounds, leave-one-log-out, every rule may pick 1 to 256 rounds:

   | Rule | Amazon | Other | All | goodreads-poetry | steam |
   |---|---:|---:|---:|---:|---:|
   | 2 rounds | 0.1656 | 0.0493 | 0.1350 | 0.0581 | 0.0334 |
   | Pooled table | 0.0248 | 0.0413 | 0.0291 | 0.0092 | 0.0365 |
   | Same-group table | 0.0165 | 0.0148 | 0.0161 | 0.0079 | 0.0070 |
   | Regression on n | 0.0235 | 0.0262 | 0.0242 | 0.0016 | 0.0198 |
   | Regression on n and group | 0.0159 | 0.0105 | 0.0145 | 0.0007 | 0.0040 |
   | Regression on n × reports per person | 0.0208 | 0.0226 | 0.0212 | 0.0052 | 0.0426 |
   | Regression on n × reports per person and group | 0.0144 | 0.0099 | 0.0132 | 0.0015 | 0.0198 |
   | Regression on n and drift | 0.0202 | 0.0290 | 0.0226 | 0.0131 | 0.0198 |
   | Regression on n × reports per person and drift | 0.0129 | 0.0122 | 0.0127 | 0.0131 | 0.0298 |

   Reports per person per week (median) are 1.3 to 2.3 on Amazon logs and 3.1 to 17.2 on 5 of the 7 other logs; goodreads-poetry (1.8) and steam (1.2) look like Amazon logs. Drift is the relative NDCG loss when the list is built from weeks 29 to 26 before the target week instead of weeks 4 to 1 before it (ties at the 100th place broken by lowest item index; `i_calculator_loo.csv`, rows `feature`). The best calculator (0.0127) beats the same-group table (0.0161) by 0.0034 and the regression on n and group (0.0145) by 0.0018, and on the two logs where the features and the label disagree it is worse than the label-based rules. Its drift feature uses the full counts of the evaluation weeks, which a server does not have. So the features act mostly as a stand-in for the domain label, and the selector (finding 1) beats all of them.

## 3. Limits and disclosures

1. **Exploratory, post hoc.** The rule was designed after the doc 13 results, on the same logs and the same period. The author chose the half-life of the average (26) before the first run and did not tune it, but no commit records this, and 26 turned out to be the best value tried at 7-day rounds (13 is slightly better at 28-day rounds; finding 3). It has not been confirmed on data that were not already used.
2. **Regret is measured against a maximum chosen with the truth.** Every rule is compared with the same noisy maximum, so differences between rules are fair, but the level of a regret is not an absolute loss.
3. **Simulated reporters.** Each round draws fresh random people. There is no recurring panel (a panel could favour long memory [inference]); the churn from switching between options is not measured; real participants could steer which option is chosen.
4. **Grids and tables.** See item 4 of section 1. The pooled and same-group tables here are not the table of doc 13 (fitted on discovery logs, 7- and 28-day rounds pooled, up to 32 rounds).
5. **D29 replay.** n = 100 only. Amazon rounds are about 3 to 16 months and the other logs' rounds range from 0.3 days to 17 months, so the "other" rows mix very different round lengths. D29's own `selfsel` arm chose among half-lives in rounds up to 8 and "never forget", used an unweighted sum and a fixed option for the first three steps; the replayed selector reads the same stored cells and uses the weighted average over the stored options (half-lives in rounds, in days, or both).
6. **Calculator.** 7-day rounds only, averaged over (log, n) cells. The drift and reports-per-person features use all weeks of the 156-week span, so a server could not have them in advance.
7. **Reviews** of the code and documents (they recomputed the tables independently and checked that no information from the future enters the choice) are same-provider subagents, not an outside audit.

## 4. What to do next (a proposal; nothing is frozen)

1. Pre-register a guarded selector (start from the table or from 2 rounds and leave it only with evidence), with criteria on the tail rather than the mean: the share of cells with regret above 0.01, the largest regret in a cell, the paired per-log difference against the same-group table, the warm-up regret over weeks 8–20, and the share of D29-replay cells worse than 2 rounds by more than 0.01. Data: other time windows of the same logs (each of the 27 logs spans at least 377 weeks from its first to its last positive event, `i_log_spans.csv`; windows are sparser, so fewer n are eligible), plus gowalla (89 weeks) and brightkite (134 weeks); no new downloads.
2. Map where the selector fails: rounds of 56 to 168 days over longer spans.
3. Differential privacy with a self-selected memory at n = 100 to 1,000 per week. The privacy results of doc 12 all use `decay:h=2` and n ≥ 1,000; their rounds range from 7 days (gowalla, 90 rounds per log) to about 11 months (ml-latest, 30 rounds), and only gowalla, brightkite and steam at 90 rounds are shorter than a month. The self-score would use only the published noisy counts.
4. A recurring panel, and the churn caused by switching between options.
5. If step 1 confirms the selector, stop tuning the memory.

## 5. Files

1. Code: `scripts/ssp/explore_selfsel_weekly.py` (needs `an_weekly.py`; the calculator block also reads the log files; the D29 replay reads `02_runs/deep.jsonl`, which is gitignored and skipped when absent).
2. Tables: `results/small_sample_popularity/03_tables/weekly/` (`i_selfsel_rules.csv`, `i_selfsel_paired.csv`, `i_selfsel_checks.csv`, `i_selfsel_d29_replay.csv`, `i_calculator_loo.csv`, `i_log_spans.csv`).
3. Rerun: `.venv/bin/python3 -W ignore scripts/ssp/explore_selfsel_weekly.py` (about 30 seconds; `--no-calculator` and `--no-d29` skip the two optional blocks).
