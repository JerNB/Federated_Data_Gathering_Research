# Design plan: tracking popularity with a few people per round

Chinese version: [`06_tracking_design_zh.md`](06_tracking_design_zh.md). Status: **plan, committed before the run.** The predictions are fixed in the docstring of `scripts/ssp/an_track.py`; the code is `scripts/ssp/track_core.py` and `run_track.py`.

## 1. The question, in one sentence

> **If you can ask only n people each round, how out of date is your popularity list compared with a full count, and how fast should you forget old answers?**

Earlier tests asked "how many people give a good list at one moment?" Real services ask every round (day, week, month). The data-gathering decisions are: people per round, and how fast to forget.

## 2. Plain-word concepts

- **Round / window.** Time is cut into 30 equal windows per log. In each window the server hears from n random active people (each reports what they chose in that window).
- **Running score.** Each item's score = the newest window's share + the old score times a forgetting factor.
- **Half-life h (in windows).** After h windows an old report counts half. Short h trusts recent reports; long h pools many rounds. Few people per round pushes h longer (less noise); fast-changing tastes push h shorter.
- **Prediction.** The top 100 by score is the prediction for the next window. Score = NDCG@100 against the whole community's choices in the next window (1 = perfect).
- **Equivalent age.** A tracker with n people per round "is as good as a full count that is L windows old". L is the equivalent age. Small L = fresh and good.

## 3. What is already known (do not claim as new)

From two literature scans (many papers were seen only in search summaries; treat as unverified until opened):

- Textbook scaling: for random-walk drift the best half-life shrinks like n^(-1/2); for steady-trend drift like n^(-1/3); in general h* ~ n^(-1/(1+b)) when squared drift over k windows grows like k^b. The exponential average is the optimal forecast for a random walk plus noise (Muth 1960; Kalman). Mazzetto 2024 (AISTATS) gives a window rule that needs no future truth.
- "Recent or decayed popularity beats all-time popularity" (Ji et al. 2020; PARE; Verachtert et al. 2022), tuned offline on full-population counts.
- Picking a smoothing constant by minimising one-step-ahead error is standard forecasting practice.

**Not found** (search-negative, not exhaustive): the best half-life as a function of people reporting per round; how many people per round track a changing top 100; which drift law real popularity logs follow.

## 4. Design

- **Data.** All 22 time-stamped logs. Window = 1/30 of the log's span (days printed next to every result). Counting unit = event rows in the window, for tracker and truth.
- **People per round.** n in 10, 30, 100, 300, 1,000, 3,000, 10,000, allowed only if n is at most 20% of the 10th-percentile number of active people per window (finite-population correction stays above 0.8). A step (window w predicts w+1) counts only if every window w-7..w+1 has at least 5n active people; a cell needs at least 10 valid steps.
- **Half-lives.** 0.25, 0.5, 1, 2, 4, 8, infinity (keep everything). 12 random draws per cell; random tie-breaks.
- **Steps.** w = 8..28 (21 steps). All half-lives are compared on the same steps.
- **Tuned half-life** (honest version): chosen on the first half of the valid steps and scored on the second half, and the reverse (cross-fitting). The in-sample best is shown only as an upper bound.
- **Self-tuned half-life** (what a server could really do): before each step pick the h whose past predictions had the smallest *share error* against the server's own sampled later windows; evaluated from the 4th valid step on. The rank-metric version (NDCG against the sampled window) is reported as a secondary.
- **Modules.** M0 ceiling: a list from half the people of the target window, scored on the other half, versus full lagged lists. M1 drift profile: noise-corrected squared share change (product of two disjoint person groups) and total-variation distance by lag. M2 tracker grid. M3 self-tuning.
- **Strata.** Amazon (16 logs, one platform, correlated drift) and Other (MovieLens 1M and latest, Gowalla, Brightkite, KuaiRand, Last.fm). A claim is "supported" only if it passes in both strata; with fewer than 3 eligible Other logs it is labelled "Amazon only".

## 5. Predictions (n = 100 people per round unless stated)

- **PA (equivalent age).** A tracker with n = 100 and h = 2 is worse than a full-community list two windows old (L_eq at least 2) in at least 2/3 of the eligible logs of each stratum.
- **PB (tuning does not matter).** The cross-fit tuned half-life does *not* beat fixed h = 2 by at least 0.01 NDCG@100 with a positive 95% lower bound, in at least 2/3 of the logs of each stratum. If PB fails, half-life choice matters.
- **PC (people per round).** The smallest n that matches a full list one window old is at least 1% of the median active people per window, in at least 2/3 of the logs of each stratum for which this is determined.
- **PD (self-tuning).** Over cells where cross-fit tuning beats fixed h = 2 by at least 0.01, the self-tuned rule recovers at least half of that gain in at least 2/3 of the cells; and it is more than 0.01 below fixed h = 2 in at most 10% of all cells (per stratum).

Reported only, not judged: h* (parabola in log h) and its slope against n; h* against the share of the next top 100 held by new items; M0 ratio; M1 slopes against the 1/2 and 1/3 theories; the people-per-round table.

## 6. Headlines by outcome

- **Flat optimum (PB true):** "Use a fixed 2-window half-life; what matters is asking every round. n people per round is worth a full count L windows old."
- **Tuning matters, selector works (PB false, PD true):** "Forget at a rate you can learn from your own reports."
- **Tuning matters, selector fails:** "The server cannot learn its own half-life; ship a default and tune offline."
- **Drift small / keep everything best:** "Ask rarely and keep everything" (likely for some non-Amazon logs).

## 7. Risks and how they are handled

- **Leakage.** Each list uses only windows up to the current one; truth is the whole community's next window, not the sampled people. The same person can be sampled in several windows; because n is at most 20% of active people and truth is community-wide, own-choice leakage is small but not zero.
- **Noisy truth for selection.** The selector uses a smooth share error against the sampled next window (not NDCG), and final accuracy is always against the full community.
- **Burn-in.** Scoring starts at window 8. Half-lives of 8 or more windows act as truncated windows early on.
- **New items.** An item first seen in the next window cannot be in any list; new-item share is reported next to accuracy.
- **Grid and slopes.** h* is read on a factor-2 grid with a parabola in log h; slopes are only reported.
- **Heavy users.** Raw event counts let heavy users weigh more, in tracker and truth alike.
- **Same-provider review.** Code is reviewed by subagents of one provider, not an outside audit.

## 8. Safety

Counts only, no model training, CPU only, 2 workers under the existing governor (fan limits 1,750/2,100 rpm, memory 8 GB, disk 10 GB free). Raw rows and run logs stay out of Git.
