# 13. Weekly rounds: how long should the memory be?

*Results of the weekly-cadence programme (11 hypotheses, 27 logs). Plan: `configs/weekly_registry.json` (W1–W7, frozen before the run) and `configs/weekly_registry_2.json` (W8–W11, written after the first results were seen). Numbers: `results/small_sample_popularity/03_tables/weekly/`.*

## 0. In one paragraph

Earlier results (doc 12) said "half-life of 2 rounds is hard to beat". That was measured with rounds of months. With **rounds of one to four weeks the best memory is much longer, and it depends on how many people answer per round**: about 32 rounds for 30 people, 16 for 100, 8 for 300, 4 for 1,000, 2 for 3,000 (only 7 logs). A one-line table built on discovery logs only, and scored on holdout logs, loses 0.018 NDCG against the best value on the grid; the old "2 rounds" loses 0.118. At 100 people per week, the hindsight grid best is about 0.60 versus 0.43 for 2 rounds; the discovery-fitted table rule, scored on holdout logs, avoids most of that loss.

**Caveat added after the first draft (finding 6, post hoc):** the table is Amazon-weighted (20 of 27 logs). On the other logs (7; 4 at n = 1,000), leave-one-log-out, a table fitted on the remaining logs is no better than 2 rounds at n = 100 and 3.3 and 4.9 times worse at n = 300 and 1,000; at n = 30 it still beats 2 rounds by about 0.09. Read the numbers above as pooled values, not as a starting point for every domain.

**Update (2026-10-16, doc 15):** on a second window the registered check K8 repeated this on the mean (non-Amazon logs, 300+ people: pooled table 0.0262 against 0.0045 for 2 rounds, 8 cells from 5 logs) but in only 2 of the 5 logs (ratebeer, steam). It is a statement about the average, not about every log.

## 1. What was done (plain words)

1. **Logs.** 27 of the 30 logs (20 Amazon, 7 other). Three logs (gowalla, brightkite, kuairand-pure) are shorter than 156 weeks and are excluded. Each log is cut to its densest 156 weeks (about 3 years). The median weekly number of active people is 4,550 (range 614 to 24,864).
2. **Rounds.** The 156 weeks are cut into rounds of 7, 14 or 28 days. Each round, a random sample of n people reports (n = 30, 100, 300, 1,000, 3,000; a log is used at n only if its median week has at least 3n active people: 27, 27, 24, 17, 7 logs).
3. **List and score.** The server publishes a top-100 list built from a decayed sum of past rounds. Quality is NDCG@100 against the full counts of the next round (1.0 = perfect; shown as plain numbers). "Half-life" is how many days or rounds it takes an old round to count half as much.
4. **Pre-registration.** W1–W7 were written and committed before any weekly run (commit `7c6fab8`). After the first results, W8–W11 were added (commit `07de0a0`) and are marked as post-hoc in design. Stage d adds new n values (30, 300, 3,000) and stage e adds the row-matched cadence. W8–W10 also reuse the already-run stage-a cells at n = 100 and 1,000; the holdout table rule is fitted on discovery logs and scored on holdout logs.
5. **Machine.** CPU only, about 20 minutes in total, 1 to 3 workers. Fans peaked at 1,492 rpm and the smoothed temperature at 64.4 °C, inside the limits (1,750 rpm, 72 °C); the raw sensor peaked at 74.1 °C.

## 2. Findings

1. **Memory table (W1, W2, W8, W9, W10). Use this instead of "2 rounds" for Amazon-like logs; finding 6 shows it is not a universal default.**

   | People per round n | Half-life in rounds (fitted on discovery logs, 7- and 28-day rounds pooled) |
   |---|---|
   | 30 | 32 (top of the options tested) |
   | 100 | 16 |
   | 300 | 8 |
   | 1,000 | 4 |
   | 3,000 | 2 (median of the best, only 7 logs; not fitted) |

   Rule of thumb: every time n grows about 3 times, halve the memory. The per-log medians of the best value are 32, 16, 8, 8, 2 at 7-day rounds and 32, 16, 6, 4, 2 at 28-day rounds for n = 30, 100, 300, 1,000, 3,000. At 100 people per week, the best half-life is at least 28 days in 27 of 27 logs (median 112 days). In 17 of 17 logs eligible at both sizes it is longer at 100 than at 1,000 people (median 224 days against 56 days). The best half-life never lengthens as n grows in every log with at least 3 sizes (median rank correlation −1.0, 24 logs).
   The count of rounds, not the number of days, stays put when the round length changes at the same n: at n = 100 the median is 16 rounds at 7, 14 and 28 days; the 7-day and 28-day values are within a factor 2 in 89% of 102 (log, n) pairs (but see limit 5: many 28-day values are cut off by the end of the grid).
   On holdout logs the table (fitted on discovery logs only) has mean regret 0.018 against 0.118 for 2 rounds (0.15 times as much; 12 to 15 holdout logs per n; per n: 0.012/0.018/0.024/0.019 against 0.149/0.171/0.106/0.047 for n = 30/100/300/1,000).
   On only the new stage-d sizes (n = 30, 300, 3,000), W8 is 49 of 58 pairs (84.5%). On the stage-d sizes with enough holdout logs for W9 (n = 30 and 300), the table regret is 0.018 against 0.128 for 2 rounds (ratio 0.14).
   Why it matters: at n = 100 per 7-day round the mean NDCG is 0.60 with the **hindsight grid-best** half-life and 0.43 with 2 rounds; at n = 1,000 it is 0.75 with the **hindsight grid-best** half-life against 0.71.

2. **The self-score also works inside one log (W5, W6).** Score last week's list on this week's reporters (needs no ground truth). Over the weeks of one log, its rank correlation with the true quality has median 0.68 (quartiles 0.56 to 0.75; 22 of 27 logs at or above 0.5; positive in all 27; discovery 0.68, holdout 0.68, clean set 0.66). Weeks whose self-score is in the lowest 10% of the log are, in 74% of cases (299 of 405 alarm weeks), also in the lowest 25% of true quality (chance is 25%). Use it as a "quality alarm" for recompute or review.

3. **Many small rounds beat few big ones, at the same number of rows (W7, W11).** Four weekly rounds of n people against one 28-day round with the same number of rows: weekly is better by +0.022 at n = 100 (95% CI +0.013 to +0.031, 27 of 27 logs) and by +0.028 at n = 300 (24 of 24). At n = 25 the gain is +0.007 (CI −0.000 to +0.014; 67% of logs; holdout logs only 53%), so it is not demonstrated at 25 people. If the monthly round is only matched on people (4n people), it wins at n = 25 (−0.012) and loses at 100 and 300 (+0.012, +0.022): the n = 25 reversal came from the monthly design collecting more rows per person.

4. **Weighting rounds by reporters helps a little, not a lot (W4).** When each week's sample is a fixed share of that week's active people, weighting by reporters gains +0.004 (CI +0.003 to +0.006, positive in 27 of 27 logs; g = 1 is best). The real week-to-week size spread is small (median p90/p10 ratio 1.5). Doc 12's +0.046 came from an invented 16-fold spread and should not be quoted for real systems.

5. **A fixed 56-day memory beats "2 rounds" at 7-day rounds by +0.121 (W3).** This is the same finding as 1 seen from the other side: 2 rounds of 7 days is only 14 days of memory. The registered wording "set memory in days, not in rounds" is not what the data say (see W8: rounds at fixed n); do not quote it.

6. **The table is Amazon-weighted (post hoc: written after findings 1–5, in neither registry).** `an_weekly_strata.py` re-reads the stored cells (no new simulation) and splits the logs into 20 Amazon and 7 other. The hindsight-best memory differs by domain at every n. Median best memory in rounds at 7-day rounds, Amazon / other: 128 / 16 at n = 30, 32 / 8 at n = 100, 16 / 4 at n = 300, 8 / 3 at n = 1,000; at 28-day rounds 32 / 16, 32 / 4, 16 / 2, 8 / 2 (`h_strata_best_rounds.csv`). Leave-one-log-out regret against each log's hindsight best (7- and 28-day rounds pooled; `h_strata_loo_regret.csv`):

   | n | Group (logs) | 2 rounds | One table fitted on all other logs | Table fitted on the same group |
   |---:|---|---:|---:|---:|
   | 30 | Amazon (20) | 0.153 | 0.012 | 0.012 |
   | 30 | other (7) | 0.104 | 0.014 | 0.007 |
   | 100 | Amazon (20) | 0.189 | 0.018 | 0.015 |
   | 100 | other (7) | 0.030 | 0.036 | 0.013 |
   | 300 | Amazon (17) | 0.123 | 0.022 | 0.019 |
   | 300 | other (7) | 0.014 | 0.047 | 0.014 |
   | 1,000 | Amazon (13) | 0.051 | 0.018 | 0.026 |
   | 1,000 | other (4) | 0.002 | 0.010 | 0.002 |

   On the other logs the pooled table is worse than "2 rounds" at n = 100 (0.036 against 0.030), n = 300 (0.047 against 0.014) and n = 1,000 (0.010 against 0.002), and clearly better only at n = 30 (0.014 against 0.104). On Amazon logs it is better at every n. A table fitted on the same group stays within 0.02 of the hindsight best except for Amazon at n = 1,000 (0.026), but it needs a domain label and rests on 7 logs (4 at n = 1,000). Do not quote "16 rounds at 100 people" as a starting point for every domain: it is the Amazon-weighted pooled value. What holds in both groups is that at 30 people a long memory beats 2 rounds by about 0.09 or more. What else the best memory depends on besides n (drift speed, reports per person) is not tested here.

## 3. Scorecard

| ID | Prediction | Result | Met |
|---|---|---|---|
| W1 | Best half-life ≥ 28 days in ≥70% of logs (7-day rounds, n = 100) | 27 of 27 (100%), median 112 days | yes |
| W2 | Longer at n = 100 than 1,000 in ≥70% of logs, strictly in ≥40% | 17 of 17 (100%), both | yes |
| W3 | 56 days beats 2 rounds by ≥ +0.01 and in ≥70% of logs (7-day rounds, n = 100) | +0.121 (CI +0.089 to +0.153), 25 of 27 logs (93%) | yes |
| W4 | Weight by reporters ≥ +0.005 and positive in ≥60% of logs (natural sizes) | +0.0042 (CI +0.0029 to +0.0059), 27 of 27 | **no** (below +0.005) |
| W5 | Median within-log rank correlation of self-score ≥ 0.5 | 0.68 | yes |
| W6 | ≥40% of lowest-10% self-score weeks are in the lowest 25% of true quality | 74% | yes |
| W7 | Weekly beats monthly by ≥ +0.02 and in ≥70% of logs (n = 100) | +0.012, 74% of logs | **no** (below +0.02) |
| W8 | Best half-life in rounds within a factor 2 at 7 and 28 days, in ≥70% of pairs | 89% of 102 pairs | yes |
| W9 | Table rule regret ≤ 0.03 and ≤ one third of 2 rounds on holdout logs | 0.018, which is 0.15 times the 0.118 of 2 rounds | yes |
| W10 | Rank correlation of log n with best half-life ≤ −0.7 in ≥70% of logs | 24 of 24 (100%) | yes |
| W11 | Weekly beats monthly, rows matched, at n = 25, 100, 300, positive in ≥60% of logs | +0.007 (67%), +0.022 (100%), +0.028 (100%) | yes (n = 25 is weak: CI includes 0; 53% on holdout logs) |

Met 9 of 11. The two misses are about size of effect, not direction.

## 4. Limits and disclosures

1. **Doc 12's "2 rounds" and this table are for different round lengths.** Doc 12 cut each log into 30 rounds (mostly Amazon rounds of 3 to 9 months), where drift between rounds is large and 2 rounds was best. This study covers rounds of 7 to 28 days. Rounds between 28 days and 3 months were not tested, so the point where the advice switches is unknown.
2. **Not a real federated deployment.** People are drawn at random from the log's active people each round; no real participation, dropout or delay.
3. **Mostly Amazon:** 20 of 27 logs, and the memory table inherits it (finding 6). On the 8-log clean set (holdout_fresh + new_holdout) W1, W3, W4, W7 (n = 100) and W11 keep their sign and the W5 median is 0.66.
4. **W8–W11 are post-hoc in design.** Stage d and stage e add new cells, but W8–W10 also reuse stage-a cells at n = 100 and 1,000. The table was fitted on 5 to 12 discovery logs and scored on 12 to 15 holdout logs.
5. **Grid effects.** Half-lives are on a grid with steps of a factor 2, so "best" is accurate to about that step. A log spans about 1,092 days, so a half-life near 900 days or more means "do not forget". At 28-day rounds the best value sits at the top of the grid in 11 of 27 logs at n = 100 (896 days) and in 9 of 27 at n = 30 (1,792 days); at 7-day rounds and n = 30 it is at 896 or 1,792 days in 11 of 27 logs. Stage-a n = 100 and 1,000 cells did not include the 1,792-day option; stage-d n = 30, 300 and 3,000 cells did. The lower grid edge also matters: 7 days is best in 3/7 n = 3,000 logs and 2/17 n = 1,000 logs at 7-day rounds, and in 1/7 n = 3,000 and 1/24 n = 300 logs at 28-day rounds. So for small n the memory in the table is a lower bound: with at most 100 people per round and rounds of weeks, forget slowly or not at all (true of the Amazon logs; on the other logs the median best at n = 100 is 4 to 8 rounds, finding 6). At n = 3,000 only 7 logs qualify, so it is not part of the rule test, and the W8 comparison is partly cut off at 28-day rounds.
6. **Cadence design.** The monthly design draws random people from the 28-day pool until its rows match the four weekly samples; this is one of several possible designs.
7. **Self-score alarm** was tested on 27 logs with the 56-day half-life only, and the 10% / 25% cut-offs were set in the registry, not tuned.
8. **Reviews** of the code and documents are same-provider subagents, not an outside audit.
9. **Seed disclosure.** Stage-c rows were produced with the original people-matched seed stream. A later code revision briefly appended `False` to that seed; the current code restores the original stream, and the stored stage-c and stage-e rows reproduce exactly.
10. **Registry-note correction.** The explanatory `frozen_note` in `weekly_registry_2.json` was corrected after commit `07de0a0` to disclose the stage-a reuse; the four W8–W11 predictions were not changed.

## 5. Files

1. Plan and predictions: `configs/weekly_registry.json`, `configs/weekly_registry_2.json`.
2. Code: `scripts/ssp/weekly_core.py` (rounds, cadence), `run_weekly.py`, `run_weekly_night.sh`, `an_weekly.py`, `an_weekly_strata.py` (post hoc, finding 6); engine additions in `deep_core.py` (`frac` sampler, `ivw` with days).
3. Tables: `results/small_sample_popularity/03_tables/weekly/` (`weekly_verdicts.json`, `a_cells.csv`, `a_regret_by_rule.csv`, `b_gain.csv`, `c_cadence.csv`, `e_within_log_selfscore.csv`, `f_best_rounds_by_round_length.csv`, `g_memory_table.csv`, `h_strata_best_rounds.csv`, `h_strata_loo_regret.csv`).
4. Rerun: `nice -n 10 caffeinate -i sh scripts/ssp/run_weekly_night.sh` (resumable; about 20 minutes), then `.venv/bin/python3 -W ignore scripts/ssp/an_weekly.py` and `.venv/bin/python3 -W ignore scripts/ssp/an_weekly_strata.py`.
