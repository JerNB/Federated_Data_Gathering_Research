# Tracking popularity with a few people per round: results

Chinese version: [`07_tracking_results_zh.md`](07_tracking_results_zh.md). Design and predictions (committed before the run): [`06_tracking_design_en.md`](06_tracking_design_en.md).
Data: 22 time-stamped logs (16 Amazon categories; MovieLens 1M and latest, Gowalla, Brightkite, KuaiRand, Last.fm). Each log is cut into 30 equal-time windows ("rounds"); in each round only n random active people report. Counts only, CPU only, 2 workers, fans stayed at idle (median 1,460 rpm).

## 1. Short answer

> **In product logs, 100 people per round give a list as good as a full count from about 1.3 to 8 rounds ago (median 4.4). Matching a full count one round old took 300 to 1,000 people per round in the 4 Amazon logs where it could be determined (300 to 10,000 with 4.5-month rounds). Forget at a fixed calendar rate, about 1.5 years for most Amazon categories (2 of our 9-month rounds); counted in rounds the rule did not carry over to other round lengths. Do not try to learn the rate from your own reports in product logs. In KuaiRand a longer memory is better and self-tuning recovered the gain in all 4 cells (the Other stratum: 9 of 12 cells, never harmful).**

The numbers behind each part are below. Only four predictions were registered in advance; their verdicts come first. Everything else is labelled exploratory or post hoc.

## 2. Terms

- **Round / window**: 1/30 of a log's time span. For Amazon it is about 9 months, for Gowalla 21 days, for KuaiRand 1 day. All "rounds" below are these windows. Real weekly rounds were not tested.
- **n**: people per round (random active people; each reports what they chose in that round).
- **Half-life h (in rounds)**: after h rounds an old report counts half.
- **Score**: NDCG@100 (1 = perfect) of the top-100 list against the whole community's choices in the next round.
- **Equivalent age**: the tracker scores the same as a full-community list that is L rounds old; L = 0 is the full count of the latest finished round, L = 1 the one before it. Smaller is better.

## 3. The four registered predictions (n = 100 people per round, h = 2 unless stated)

Verdicts need both strata (Amazon, Other) to pass; with fewer than 3 eligible Other logs the label is "Amazon only". Eligible Other logs at n = 100: 3 (Brightkite, KuaiRand, MovieLens latest).

| Prediction | Verdict | Amazon | Other |
| --- | --- | --- | --- |
| **PA** 100 people per round is worse than a full count 2 rounds old | **Supported** | 11 of 14 logs (exceptions: electronics, movies-tv, video games, at 1.3 to 1.8 rounds) | 2 of 3 (MovieLens latest at 1.7) |
| **PB** tuning the half-life does not matter (tuned beats h = 2 by less than 0.01) | **Mixed** | **14 of 14: does not matter** | 1 of 3 (KuaiRand +0.077, MovieLens latest +0.016: tuning helps) |
| **PC** matching a full count one round old needs at least 1% of active people | **Mixed** | 1 of 4 determined (books 0.98%, electronics 0.38%, movies-tv 0.51% are below 1%; video games 2.5%) | 4 of 4 |
| **PD** the server can learn its own half-life | **Mixed** | **No**: the self-tuned list is more than 0.01 worse than h = 2 in 51% of 59 cells | **Yes**: recovers at least half of the tuning gain in 9 of 12 cells; harmful in 0 of 16 |

Plain reading: in the Amazon product logs a fixed half-life of about 2 rounds is already as good as tuning, and self-tuning backfires. In KuaiRand a longer memory is better and self-tuning recovered the tuning gain in all 4 cells; in Brightkite 1 of 2 cells (at n = 100 its tuning gain, +0.057, has a 95% lower bound just below 0, so PB counts it as "does not matter"). MovieLens latest is different (best half-life 1 round at n = 100, 0.5 at larger n), with small tuning gains. The PB "Other" label rests on bootstrap lower bounds near 0 (MovieLens latest +0.003); it was the same under three seeds.

## 4. What the numbers mean for a service

1. **People per round.** The smallest n that matches a full count one round old (n_L1), where the grid reached it at 30 windows: books 1,000 (about 1% of active people), electronics 1,000 (0.4%), movies-tv 300 (0.5%), video games 300 (2.5%). For most other logs the grid stopped before reaching it. With 4.5-month rounds (extension E2) n_L1 is determined for 10 of 16 Amazon logs: 300 (movies-tv, software, video games), 1,000 (books), 3,000 (electronics) and 10,000 (health, pet supplies, sports; 33 to 42% of active people). In books, electronics and movies-tv n_L1 changed less than the share of active people, which hints that a count of people matters more than a percentage, but the three 10,000 cases do not fit a fixed count.
2. **Half-life.** At n = 100 the best fixed half-life is 2 rounds in 13 of 14 Amazon logs with a result (4 in toys), with no gain from tuning. The textbook says the best half-life should fall with n like n to the minus one third or one half. We measured a median slope of **-0.13 (Amazon, 12 logs)** and -0.29 (Other, 2 logs): the best half-life hardly depends on n. This goes with how fast the product catalogue turns over (see 3), not with sampling noise. Other logs at n = 100: Brightkite 8 rounds, KuaiRand no forgetting, MovieLens latest 1 round.
3. **New items go with it.** Across Amazon logs the best half-life at n = 100 falls as the share of the next top 100 held by new items rises (Spearman -0.69, 14 logs). All logs together: -0.41 (16 logs). This is a correlation, not a tested cause.
4. **Drift, not noise, is the limit.** A list built from half the people of the target round and scored on the other half reaches 0.44 to 1.0 (so sampling noise is small). A full list from the previous round reaches only 31% (movies-tv) to 106% (MovieLens 1M, whose timestamps are concentrated) of that level; median 85%; in books it is 42%.
5. **Drift shape (exploratory).** The noise-corrected squared change of item shares grows with lag with a log-log slope of 0.2 to 1.26 (median 0.72); total variation 0.28 to 0.77 (median 0.58). Nothing is near 2, so popularity is not a steady trend; it behaves between saturating and random-walk.

## 5. Extensions (post hoc; declared before running) and the robustness prediction

- **E1 (more n)**: eligibility by step validity only: 88 cells instead of 75. PA, PB, PC, PD labels unchanged (PA supported, PB mixed, PC mixed, PD mixed).
- **E2 (window width)**: 60 windows (4.5 months for Amazon) and 20 windows (14 months).

| Label | 30 windows (E1) | 60 windows | 20 windows (12 logs only) |
| --- | --- | --- | --- |
| PA | supported | supported | not supported |
| PB | mixed | **not supported** (tuning matters in 9 of 16 Amazon logs) | mixed |
| PC | mixed | **supported** | mixed |
| PD | mixed | mixed | not supported |

**Robustness prediction R failed**: at most 2 of 4 labels stay the same at each width (needed 3). The "h = 2 is enough" rule does not transfer across window widths as stated; it is a calendar rule that has to be re-read in days.

- **Calendar half-life (post hoc, exploratory).** Converting each log's grid-best half-life to days: 11 of 18 logs have the same best half-life in days at every width they were run at (only 30 and 60 windows): 10 ordinary Amazon categories at 494 to 587 days (about 1.5 years) and Gowalla at 83 days. The other 7 differ by at most one grid step for 6 of them (books x1.5; electronics, movies-tv, toys, video games, Brightkite x2), which the factor-2 grid cannot separate; only MovieLens latest differs more (x3). So the best half-life is roughly constant in calendar time within the resolution of the grid.

## 6. What this does not show

- Windows are long (days for KuaiRand, months for Amazon) because the logs are sparse. Weekly or daily rounds at Amazon scale were not tested.
- n above 100 to 300 is available only for some logs (early windows are thin, so the validity rule drops large n); n_L1 is undetermined for most logs.
- Only the 3 Other logs eligible at n = 100 (Brightkite, KuaiRand, MovieLens latest) can speak for slow-changing logs; the Other-stratum verdicts rest on 3 to 5 logs.
- The three drift laws (random walk, steady trend) were not fitted to tracker behaviour; M1 reports log-log slopes only.
- Raw event counts, so heavy users weigh more; the same person can be sampled in several rounds.
- Reviews were by subagents of one provider (a code review and an execution-probe check found only analysis-script issues: the PD "Amazon only" rule, the n for 80% report, NaN-safe JSON; a later reviewer found the number and wording errors corrected in this document). The analysis-script fixes were made while the run was in progress and committed together with the results, so "fixed before the results were read" cannot be checked from the history. The PD docstring wording was also clarified after registration; the rule in the code did not change. Not an outside audit.

## 7. Files

- Design and registered predictions: `docs/small_sample/06_tracking_design_{en,zh}.md`.
- Code: `scripts/ssp/track_core.py`, `run_track.py`, `an_track.py`, `an_track_calendar.py`.
- Tables (`results/small_sample_popularity/03_tables/`): `an_track_verdicts.json` (registered verdicts), `an_track_cells.csv`, `an_track_calculator.csv`, `an_track_m0m1.csv`; extensions `*_w30_steps`, `*_w60_steps`, `*_w20_steps`; `an_track_calendar.csv`.
- Figures: `04_figures/an_track_main.png` (equivalent age, tuning gain, three methods at n = 100), `an_track_calendar.png`.
- Raw run rows (`02_runs/track.jsonl`, 312 rows, about 20 MB) stay out of Git.
