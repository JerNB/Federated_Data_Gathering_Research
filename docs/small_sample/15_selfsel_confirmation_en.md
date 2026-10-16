# 15. Does the self-score selector still work on a new time window? (pre-registered confirmation)

*Pre-registered: the plan `configs/selfsel_confirm_registry.json` (13 hypotheses K1–K13, thresholds, decision rule) was committed in `8a0ef99` before any cell was run. Verdicts are computed by `scripts/ssp/an_confirm.py`. Anything marked "post hoc" was not in the plan. Tables: `results/small_sample_popularity/03_tables/confirm/`.*

## 0. Summary

1. **Decision: CONFIRMED** (the four registered conditions K1–K4 are met). On a second, adjacent 156-week window of the same 27 logs, at 7-day rounds, the self-score selector again had lower regret than tables fitted leave-one-log-out: −0.0103 NDCG against the same-group table (95% interval −0.0193 to −0.0039), −0.0185 against the pooled table (−0.0283 to −0.0104). Both log groups are negative. The edge is smaller than on the window where the selector was designed (−0.0141).
2. **Thin tail.** Only 2 of 90 cells have a regret above 0.01 (both Amazon categories at 30 people; largest 0.0104). The selector was never more than 0.009 worse than the better of "2 rounds" and the same-group table.
3. **Three registered claims were not met or came out reversed.**
   - K12 (re-picking every week beats the memory that was best on the same log's previous window): reversed. The fixed memory from the log's own history was better by 0.0011 (+0.0001 to +0.0020), mostly at 30 people.
   - K11 (two new short check-in logs): the selector beat the table in 5 of 7 cells, 6 of 7 were required.
   - K13 (warm-up): almost a tie, 0.0056 against 0.0059.
4. **At 30 people the advantage over the same-group table is not established** (−0.0021, interval −0.0065 to +0.0014; post hoc split). From 100 people up it is clear (−0.0140, interval −0.0246 to −0.0061).
5. **Monthly rounds repeat the earlier picture**: better than the same-group table at 100+ people (−0.0060), worse at 30 people (+0.0081; the selector is worse in 81% of logs), as registered.
6. **The best memory of a (log, people-per-round) cell hardly moves between the two windows**: the same half-life in 77% of cells, within a factor of 2 in 98%, a factor of 8 or more in none (post hoc). That is why a table fitted on the old window still works (K5–K7) and why a log's own history is the strongest fixed rule.
7. **Reading for practice (post hoc, not a registered claim):** with no history for a log and 100+ people per weekly round, use the selector. With a year or more of the log's own history, fix the memory from it. At 30 people per round, or at monthly rounds with 30 people, use a memory table or the log's own history; the selector adds little or loses.

## 1. What was done

1. **Same engine, new data.** Every log was cut into a fresh 156-week window that does not overlap the window of the weekly study (the densest free side of the log). Two short check-in logs (gowalla 89 weeks, brightkite 134 weeks) were used whole. 187 cells in total: 90 at 7-day rounds and 90 at 28-day rounds on the 27 logs, 7 on the two short logs. A cell is one log, one number of people per round (30, 100, 300 or 1,000) and one round length; 8 random draws; every round a fresh random sample of the active people reports all their rows.
2. **Nothing was tuned.** The selector is the one of doc 14 (publish the memory whose self-scores, averaged with weights that halve every 26 rounds, are best; before any score exists, publish the option nearest to 2 rounds). Tables are the memory in rounds that is best on average over the other logs of the same window (leave one log out). Regret is the best fixed memory of the same cell, chosen with the truth, minus the value of the rule; smaller is better.
3. **Pre-registration.** Thresholds were set from the exploration numbers with a safety margin and checked by a critic review before the freeze. The decision rule was fixed: CONFIRMED if K1–K4 are all met; WEAKENED if the K1 interval is below 0 but something else in K1–K4 fails; NOT CONFIRMED otherwise.
4. **An interruption, disclosed.** At 00:08 the Mac ran out of swap space and macOS killed system services, including the ones that grant access to `~/Documents`. The run lost its files: 102 of 187 cells were done, 85 never ran, and 86 cell keys logged `PermissionError` (one of them succeeded on its retry). After a restart the same resumable command ran the remaining 85 cells (01:04–01:11). The failed-attempt log was moved aside, not deleted. No cell was dropped. Two cells (one from each invocation) were recomputed independently and match exactly.

## 2. Results (registered hypotheses)

Regret differences are paired per log (each log's cells averaged; 27 logs; 95% bootstrap interval over logs). Negative means the selector is better.

| ID | Registered claim (plain words) | Observed | Verdict |
|---|---|---|---|
| K1 | 7-day: selector beats the same-group table by at least 0.005, interval below 0, both groups | −0.0103 (−0.0193 to −0.0039); Amazon −0.0117, other −0.0064; 3 cells without a same-group table | MET |
| K2 | 7-day: selector beats the pooled table | −0.0185 (−0.0283 to −0.0104) | MET |
| K3 | 7-day: at most 10% of cells with regret above 0.01, none above 0.03 | 2.2% (2 of 90); largest 0.0104 | MET |
| K4 | 7-day: at most 5% of cells where the selector is worse than the better of 2 rounds and the same-group table by more than 0.01 | 0%; largest excess 0.0090 | MET |
| K5 | selector beats a same-group table fitted on the old window | −0.0102 (−0.0190 to −0.0035) | MET |
| K6 | selector beats a pooled table fitted on the old window | −0.0203 (−0.0294 to −0.0124) | MET |
| K7 | the old pooled table keeps its edge over 2 rounds (regret ≤ 0.04 and ≤ ⅓ of 2 rounds) | 0.0220 against 0.1404 (ratio 0.157) | MET |
| K8 | non-Amazon logs, 300+ people: the pooled table is at least twice as bad as 2 rounds (gap ≥ 0.005) | 0.0262 against 0.0045 (×5.8, gap 0.0217); 8 cells from 5 logs. Only 2 of the 5 logs show it | MET (on the mean) |
| K9 | 28-day rounds, 100+ people: selector beats the same-group table | −0.0060 (−0.0124 to −0.0011) | MET |
| K10 | 28-day rounds, 30 people: the same-group table is better (a negative result registered in advance) | selector +0.0081 (+0.0033 to +0.0132); worse in 81% of logs | MET |
| K11 | two new short logs: selector ≤ same-group table in at least 6 of 7 cells, on average in each log, never harmful | 5 of 7 cells (71%); both log averages negative (brightkite −0.0463, gowalla −0.0302); no cell above 2 rounds + 0.01 | NOT MET |
| K12 | re-picking every week beats the memory that was best on the same log's previous window | selector +0.0011 (+0.0001 to +0.0020): the old-window memory was better; at 30 people +0.0038 (+0.0021 to +0.0054); at 100+ people −0.00004 (−0.0013 to +0.0009) | NOT MET (reversed) |
| K13 | warm-up (weeks 8–19, reported only): selector better than the pooled table | 0.0056 against 0.0059 | MET (a tie in practice) |

Decision: **CONFIRMED** (K1–K4).

## 3. Interpretation

1. **The selector's edge replicates, at a smaller size.** The drop from −0.0141 to −0.0103 is the shrinkage one expects when a rule was designed on the first window. It is still clearly below zero, in both log groups, and against both a table fitted on the same period and a table fitted on the past (K5, K6: a table made in advance is what one would deploy).
2. **It matters most from 100 people per weekly round.** There it beats the same-group table by 0.014 on average (post hoc split, interval −0.0246 to −0.0061). At 30 people the table is as good (−0.0021, interval includes 0), and a 156-week history of the same log is better than the selector (K12 split: +0.0038).
3. **The selector is a cold-start tool.** Its job is to find the memory when nobody knows it. Once a log has a long history, its best memory can be fixed and it hardly moves between windows (77% identical, 98% within a factor of 2). So the registered hope "re-picking every week beats the log's own old choice" is not supported: it is a tie from 100 people up and a loss at 30 people. The fair claim is only "better than tables made from other logs".
4. **No heavy tail at weekly rounds, a heavy tail at monthly rounds with tiny samples (post hoc).** At 28-day rounds, 17 of 90 cells have regret above 0.01, 6 above 0.02 and 3 above 0.03 (all Amazon at 30 people: patio-lawn-and-garden 0.044, automotive 0.036, health-and-household 0.032; largest 0.0436). There are only 39 rounds to learn from and every round holds 30 people. This is the region where doc 14 already warned.
5. **Domains differ.** On the non-Amazon logs the selector's edge over the same-group table is smaller (−0.0064) but consistent (86% of logs). A pooled table is a poor choice there at 300+ people: 5.8 times the regret of 2 rounds on average (K8), although it was worse than 2 rounds in only 2 of the 5 logs (ratebeer, steam). Use the selector or a same-group table instead.
6. **Near tie in warm-up.** In weeks 8–19 the selector is only 0.0003 better than the pooled table (window 1: 0.0045 better). So "it pays off quickly" is not supported; the selector's edge comes later.
7. **What the selector picks, in rounds (post hoc).** Medians of the selector's own choice are the same as the best fixed memory in hindsight: Amazon 64, 32, 16, 8 rounds at 30, 100, 300, 1,000 people; other logs 8, 4, 4, 2 rounds (best fixed: 8, 4, 4, 4). These are a sensible default table when no selector is run.

## 4. Post-hoc extras (not registered)

Written by `scripts/ssp/explore_confirm_extra.py`; rows of `k_paired.csv` for the scopes "w2 amazon", "w2 other", "w2 n>=100", "w2 n=30" are also post hoc.

1. **Mean regret on the old and new windows (7-day / 28-day rounds, 95 and 90 cells):**

   | Rule | Old window 7-day | New window 7-day | Old window 28-day | New window 28-day |
   |---|---:|---:|---:|---:|
   | 2 rounds | 0.1350 | 0.1404 | 0.0889 | 0.0944 |
   | Pooled table | 0.0291 | 0.0210 | 0.0164 | 0.0121 |
   | Same-group table | 0.0161 | 0.0133 | 0.0124 | 0.0092 |
   | Selector | 0.0027 | 0.0024 | 0.0075 | 0.0069 |

2. **Stability of the best memory (7-day):** same best half-life in 77% of cells (Amazon 78%, other 73%), within a factor of 2 in 98%, in none off by a factor of 8 or more. At 28-day rounds: 62% and 94%.
3. **Selector against the same-group table by n (7-day, all logs, paired):** 30 people −0.0021 (−0.0065 to +0.0014); 100 or more −0.0140 (−0.0246 to −0.0061). At 28-day rounds over all n: −0.0018 (−0.0073 to +0.0024), so the monthly-round advantage is established only for n ≥ 100 (K9).

## 5. Limits and disclosures

1. **Not an independent sample of the world.** The same 27 logs (20 are Amazon categories from one platform) and a window next to the old one; the intervals over logs treat logs as independent, which overstates independence. Only gowalla and brightkite are new logs, and they are short (K11).
2. **Simulated reporters.** Every round draws fresh random people; there are no real participants, no dropout, no volunteers and no privacy noise (doc 16 tests some of these).
3. **Where the licence ends.** The result may be stated only as: on a second, adjacent 156-week span of the same 27 logs (20 Amazon), with simulated random reporters, at 7-day rounds and 30 to 1,000 people per round, the self-score selector again had lower regret than tables fitted leave-one-log-out. It does not cover new domains, rounds longer than 28 days, fewer than about 40 rounds, real participants, or fixed memories chosen on the log's own history (K12 went the other way).
4. **K13 bin.** The registry text says steps 8–20; the code uses steps 8–19 (half-open bin). With step 20 included the numbers are 0.0054 and 0.0061, still met.
5. **Cells without a same-group table.** Only 3 non-Amazon logs are eligible at 1,000 people (the three Goodreads logs); with one left out there is no same-group table for 3 cells at each round length. They drop out of K1 and K9 and K4 falls back to 2 rounds there.
6. **After the freeze, only the run governor changed** (an absolute swap-space limit in `scripts/ssp/quiet.py`, `configs/ssp_quiet.json`; committed separately). It decides when workers pause, not any number. The hash block in `k_verdicts.json` covers the registry and the confirmation scripts but not the simulation code; `git diff` shows `deep_core.py`, `track_core.py` and `weekly_core.py` unchanged since the freeze.
7. **Edge cases left unfixed on purpose.** A prober found that `confirm_core.py` raises on an empty log or a log with no positive events, and accepts `round_days` of 0, −1 or 1e9. None can occur with the registered logs and round lengths; the file is hash-frozen, so it was not edited.
8. **Reviews are same-provider subagents, not an outside audit.** An Opus reviewer re-implemented the analysis from the raw rows without importing the project code and matched every stored value to 1e-5 (0 mismatches), checked the freeze against git, and recomputed two cells exactly. A verifier reran the analysis and the self-test (byte-identical outputs, self-test passes). The reviewer's first run was cancelled by a user interruption and was repeated once.

## 6. What comes next

1. Stress tests of the selector under conditions the logs cannot show: volunteers who report with unequal chances, an unbiased audit sample, privacy noise, and the churn of the published list (doc 16).
2. A guarded version for the weak regions (30 people, monthly rounds), designed and registered on its own fresh data; this registry forbids designing it on these windows.
3. Materials and sources used so far are recorded in the materials ledger (doc 90).

## 7. Files

1. Plan and code: `configs/selfsel_confirm_registry.json`, `scripts/ssp/confirm_core.py`, `run_confirm.py`, `an_confirm.py`; post hoc: `explore_confirm_extra.py`.
2. Tables: `results/small_sample_popularity/03_tables/confirm/` (`k_verdicts.json` with the file hashes, `k_cells.csv`, `k_summary.csv`, `k_paired.csv`, `k_tails.csv`, `k_logs.csv`; post hoc: `l_pooled_windows.csv`, `l_best_arm_stability.csv`, `l_selector_picks.csv`).
3. Raw rows (not committed): `results/small_sample_popularity/02_runs/confirm.jsonl`, `confirm_logs.json`; failed attempts: `confirm.jsonl.errors.failed-2026-10-16`.
4. Rerun: `nice -n 10 caffeinate -i .venv/bin/python3 -W ignore scripts/ssp/run_confirm.py`, then `.venv/bin/python3 -W ignore scripts/ssp/an_confirm.py` (self-test: `--selftest`) and `explore_confirm_extra.py`.
