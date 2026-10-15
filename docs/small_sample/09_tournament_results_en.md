# 09. Hypothesis tournament: results

## 1. What was run and what came out

Three rounds of the tournament (design: document 08) tested 12 candidate ideas plus two offline checks (H7, H16) and one calendar check (H8). All six ideas of round 1 were killed at the first screen. In round 2, three of four were killed at screen and one failed (batching loses 0.016 at n = 300; at n = 30 and 100 the loss is inside the margin but not proven). In round 3, one idea (forget in days, not in rounds) beat the old rule by a large margin on the holdout logs but did not pass the strict test on the discovery logs. Nothing was left unchecked: every candidate, including the dropped ones, is in the ledger below.

The whole tournament took about 25 minutes of computer time, not a night. The time limit is ideas, not computing. More rounds need new candidate ideas, and each new round counts against the same error budget.

## 2. Scoreboard

Differences are in NDCG@100 against the baseline (sum of rounds with a half-life of 2 rounds). Negative means worse. "Discovery" = 10 logs, 8 of them Amazon. Numbers are for n = 100 people per round unless stated.

| # | Idea in plain words | Result | Key number |
| --- | --- | --- | --- |
| H1 | Use only the last 3 rounds | Killed at screen | -0.039 (last 2 rounds: -0.063; last 5: -0.025) |
| H2 | Extra weight for rising items | Killed at screen | -0.039 (half weight: -0.018) |
| H3 | Half recent, half all-time | Killed at screen | -0.082 |
| H4 | Each person reports at most 3 rows | Killed at screen | -0.050 (1 row: -0.112) |
| H5 | Ask in batches (every second round) instead of every round | Killed at screen as "worse"; the difference is about zero | +0.003 at n=100 |
| H6 | Drop items reported once in a round | Killed at screen | -0.064 (threshold 3: -0.167) |
| H9 | Batching loses less than 0.01 | Failed on discovery logs (n=300 is outside the margin; n=30 and 100 are inside it but not proven) | n=30: +0.001; n=100: -0.0004; n=300: -0.016; every 4th round: -0.007 / -0.014 / -0.053 at n=30 / 100 / 300 |
| H10 | Each person reports at most 10 rows | Killed at screen | -0.011 (30 rows: -0.0006) |
| H11 | Publish only items with a running count of 3 or more | Killed at screen | -0.194 (count 5: -0.253) |
| H12 | Damp heavy hitters (square root of shares) | Killed at screen | -0.012 |
| H14 | Forget over 534 days, not over 2 rounds | Not proven on discovery logs; holdout passes at 60 and 90 rounds | holdout, n=100: +0.024 at 60 rounds, +0.054 at 90 rounds |
| H15 | Forgetting in days never loses more than 0.01 | Confirmed at 30 rounds only (n = 30 and 100). Nearly true by construction: 534 days is about 2 rounds at 30 rounds | +0.0002 at n=100, 30 rounds |
| H7 | Pick the half-life from the share of new items | Not met | rule worse in 4 of 19 logs, better in none; regret 0.0152 vs 0.0105 (p = 0.14) |
| H8 | Best half-life is a calendar length | Met by the registered rule, with weak support | (a) 9 of 11 logs at 90 rounds (6 of 11 at 45 rounds, which cannot separate the two rules); (b) 3 of 4 new Amazon categories within 0.005 of the best, but at 30 rounds the calendar pick equals 2 rounds in all 4, so (b) does not separate the rules |
| H16 | Drift speed predicts the best half-life | Not met | rank correlation 0.49 (needed 0.7) over 86 log-and-round-length cells |

## 3. What this means in practice

1. **Keep the baseline unless you have a calendar reason to change it.** Five different ways of weighting rounds (last 3, rising items, recent plus all-time, square root of shares, threshold on counts) were all worse than a decayed sum with a half-life of 2 rounds. None beat it on average on the discovery logs.
2. **If your rounds are much shorter than 9 months, a half-life in days looked better.** On Amazon-type logs, when the log was cut into 60 or 90 rounds (about 3 to 4.5 months each, instead of 9), a half-life of 534 days beat "2 rounds" on the holdout logs by 0.018 to 0.054 NDCG (400 days: 0.012 to 0.057). The effect is on average large but not uniform: it was positive in 75% to 83% of holdout logs (400 days: 83% to 92%), and some logs lost up to 0.03. At 30 rounds the two rules are the same within 0.002. At 45 rounds the gain (+0.007) was below the 0.01 margin, and rounds longer than 9 months (20 rounds) could not be tested (2 to 3 logs). A half-life of 700 days gained less (+0.010 at 60 rounds, +0.031 at 90 rounds) and was less reliable. This is not confirmed under the rule I wrote in advance (section 4).
3. **Privacy costs a lot at 100 people per round.** Capping what a person can report loses 0.011 at 10 rows, 0.050 at 3 rows and 0.112 at 1 row. Dropping rarely reported items loses 0.06 to 0.25, because most items in the top 100 were reported only a few times. A cap of 30 rows is nearly free (-0.0006).
4. **Asking in batches is roughly free up to every second round for 100 people or fewer.** The loss is about zero at n = 30 and 100 but becomes 0.016 at n = 300, and asking every fourth round loses 0.007 to 0.053 (n = 30 to 300). The n = 300 loss is outside the margin, and with 10 discovery logs the interval at n = 30 and 100 is too wide to prove "less than 0.01", so the pre-registered claim failed and "harmless" is not shown.
5. **There is no simple measured number that tells you the best half-life for other kinds of logs.** The share of new items (H7) and the speed at which the whole community's counts drift apart (H16) both failed. For non-Amazon logs the earlier tracking study found that letting the server tune its own half-life recovered at least half of the possible gain in 9 of 12 cells and never hurt.

## 4. Details that change how far to trust each result

- **Discovery versus holdout.** H14 had a mean gain above 0.01 on the 8 Amazon discovery logs at 60 and 90 rounds (+0.021 and +0.041), but the strict interval (error budget shared among 12 hypotheses and 10 round-length-and-n combinations) did not exclude 0. On the holdout logs (12 at n = 30, 11 at n = 100, including 5 and 4 new categories), the same rule passes at 60 and 90 rounds (it was run on the holdout logs only because the rule is shared with H15, which passed discovery). By the rule I wrote in advance this is not a confirmation, because both stages must pass.
- **Grid edge.** In the 90-round runs the best half-life was 4 rounds (about 330 to 440 days) in 11 of 19 logs and 8 rounds (the largest grid value) in 4 logs. For the 10 Amazon logs flagged in the earlier study, the best half-lives in days at 90 rounds are about 1.5 times shorter than at 30 and 60 rounds (330 to 390 days against 494 to 587). The calendar idea holds approximately, not exactly.
- **H8 is partly circular.** The 534-day number came from the same earlier study on the older logs. Only the 4 new categories (industrial was too small) are blind to it: 3 of the 4 were within 0.005 of the best half-life, but at 30 rounds the 534-day pick equals 2 rounds there, so this part cannot tell a calendar length from a rounds rule.
- **The first run of round 1 was seen before a fix.** It had a bug that gave the batching rule up to 11% (every second round) and 15% (every fourth round) fewer people in 12 of 20 cells (mean shortfall 2.2% and 3.4%). All six ideas were killed in that run too. After the fix and a complete rerun the results above are from the corrected code. A smaller shortfall remains by design (unspent budget cannot be recovered in the last round): batching gets about 2% (every second round) and 4% to 6% (every fourth round) fewer people than the baseline, which slightly favours the baseline. Round 2 and 3 were designed after seeing earlier rounds' discovery results; the holdout logs were never used to evaluate any rule before the final stage of its round. The family size was increased accordingly (10 hypotheses for round 2, 12 for round 3), so each test is stricter.
- **The registered error level and the applied one differ for round 3.** The registry text (written for round 1) says each n is tested at level 0.05 / (family x 3) and 0.05 / 3 on the holdout. The code divides by the number of (round length, n) combinations, which is 10 in round 3, so it applied 0.05 / (12 x 10) and 0.05 / 10. Under the text as registered, H15 would also pass at 45 rounds with n = 30; H14 would still not pass on the discovery logs. The code was frozen in the same commit as the round-3 registry; I did not amend the registry text.
- **H7 is not a blind test.** The original criterion was found unreachable after the numbers had already been seen, so I changed it to mean regret and report both.
- **Amazon dominates.** 8 of 10 discovery logs and 12 of 14 holdout logs are Amazon, so every confirmation means "confirmed on Amazon-dominated logs".
- **Long rounds.** At 30 rounds, an Amazon round is about 9 months. Shorter rounds were only tested through the 45, 60 and 90 round runs.
- **Code edit after seeing results.** `an_tourney.py` was edited after the round-3 holdout analysis crashed on sorting mixed ids; the same edit added the wording of one status line. No gate rule or number changed.
- **Reviews.** An Opus critic and three Opus code reviews found defects (runner crash, averaging over n, a silently skipped stage, wrong ledger rows, unpaired ties, a mismatched batching budget, reused draws) that were fixed before results were used. The later changes (multi-width gates, ledger wording, the H16 script) had a separate review only for the documents. These are same-provider subagents, not an outside audit.

## 5. Files

- Registries (frozen before each round): `configs/tourney_registry.json`, `tourney_registry2.json`, `tourney_registry3.json`.
- Code: `scripts/ssp/tourney_core.py`, `run_tourney.py`, `an_tourney.py`, `an_drift_rule.py`, `run_tourney_night.sh`.
- Tables: `results/small_sample_popularity/03_tables/tourney_*.csv`, `tourney2_*.csv`, `tourney3_*.csv`, `tourney_cal.csv`, `tourney_cal_fresh.csv`, `tourney_offline_h7.csv`, `tourney_drift_rule.csv`, and the ledgers `tourney_ledger.csv`, `tourney2_ledger.csv`, `tourney3_ledger.csv`.
- Raw rows (gitignored): `02_runs/tourney*.jsonl`. The first run with the cadence bug is kept as `tourney_firstrun.jsonl`, `tourney_screen_firstrun.csv`.
- Rerun: `nice -n 10 caffeinate -i sh scripts/ssp/run_tourney_night.sh`; for round 2 or 3 set `TOURNEY_GEN=2` or `3`.
