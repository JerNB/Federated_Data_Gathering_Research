# 08. Hypothesis tournament: design (written and committed before the run)

## 1. Why this exists

Each study so far tested a few ideas by hand. Most failed or turned out obvious. A tournament does the same job in bulk and with fixed rules: many candidate ideas are tried on the same data, weak ones are dropped early, and only the ones that survive tests on logs they were never tuned on are reported as useful. It works like a small machine-learning pipeline: candidates, a training set (discovery logs), a held-out test set (holdout logs), and a record of everything tried, including the failures.

It cannot invent ideas by itself. I add new candidates between runs. It decides, without me, which candidates are worth keeping.

## 2. The setting (unchanged from document 06)

A log is cut into 30 equal-time rounds. Each round, n random active people report what they did. The server builds a top-100 list from the reports so far. The list is scored (NDCG@100, 1 is perfect) against what the whole community did in the next round. Every candidate rule sees exactly the same reporters, so differences between rules are paired. The baseline is the rule recommended in document 07: add up each round's shares with a half-life of 2 rounds (`decay:h=2`).

## 3. The six candidates and what I predicted

| # | Idea in plain words | Rule tested | Type of claim | My prediction (written before the run) |
| --- | --- | --- | --- | --- |
| H1 | Use only the last 3 rounds, no gradual forgetting | `slide:K=3` | beats the baseline by at least 0.01 | fails: a hard cut-off throws away useful answers |
| H2 | Give extra weight to items that are rising | `mom:beta=1` | beats the baseline by at least 0.01 | helps only where many new items enter the top 100; on Amazon about zero |
| H3 | Half recent answers, half all-time answers | `blend` | beats the baseline by at least 0.01 | fails on Amazon, may help elsewhere |
| H4 | Each person may report at most 3 rows per round (privacy, bandwidth) | `cap:c=3` | loses less than 0.01 | holds |
| H5 | Ask twice as many people every second round instead of n every round | `cad:m=2` | is worse by 0.01 or more | holds: old answers hurt more than small samples |
| H6 | Drop items that were reported only once in a round (privacy) | `thresh:t=2` | loses less than 0.01 | holds |

Each idea also has one or two secondary variants (for example `cap:c=1`). Only the primary rule decides. Secondary variants are reported but cannot rescue a hypothesis.

Two more checks use existing or new data without a screening stage:

- **H7** (offline): choose the half-life from how many new items enter the top 100, with the line fitted on the other logs. Prediction: it does not beat a fixed 2 rounds. The first wording ("better in 70% of logs") was found unreachable and was changed to mean regret after the numbers had already appeared in a test run, so H7 is not a blind test. The original and the changed result are both reported.
- **H8** (calendar length): the best half-life is a fixed time span in days, not a fixed number of rounds. Test (a): at 90 rounds, in logs that earlier looked the same in days, the half-life predicted by days must beat the half-life predicted by rounds in 70% of the logs where the two predictions differ. Test (b): on 5 new Amazon categories, the half-life nearest 534 days must be within 0.005 of the best half-life in 70% of logs.

## 4. Stages and pass rules (fixed in `configs/tourney_registry.json`)

1. **Screen.** Discovery logs, n = 30 and 100, 6 draws. Keep a hypothesis if its main rule moves in the predicted direction at either n. Everything else is dropped and recorded as dropped.
2. **Full.** Discovery logs, n = 30, 100 and 300 (where enough people are active), 12 draws. Each n is judged separately, because the average over n can hide a loss at small n. PASS rules:
   - beats the baseline: average gain at least 0.01, positive in at least 70% of logs, and a one-sided t bound above 0 at level 0.05 / (6 hypotheses x 3 values of n);
   - loses little: the t bound of the average difference is above -0.01 and no single log is worse than -0.03;
   - worse than the baseline: average at most -0.01, negative in at least 70% of logs, upper bound below 0.
   At least 4 logs must be eligible at an n; otherwise that n is "not testable".
3. **Holdout.** Logs not used in stages 1 and 2: 7 older Amazon logs, 2 other logs, and 5 new Amazon categories. The same rules apply at level 0.05 / 3, and the average on the 5 new categories alone must have the same sign. A hypothesis is CONFIRMED only at the values of n that pass both stage 2 and stage 3.
4. **Missing results.** If any planned cell is missing, the stage is marked INCOMPLETE. It is never counted as a failure.

## 5. The split of logs

Fixed before any run, by sorting log names with a fixed hash: discovery = 8 Amazon + 2 other logs (arts-crafts, books, electronics, musical-instruments, office-products, patio-lawn-garden, pet-supplies, video-games, gowalla, ml-latest). Holdout = 7 Amazon + 2 other (automotive, cell-phones, grocery, health, movies-tv, sports, toys, brightkite, kuairand-pure) plus 5 new Amazon categories (baby, cds-and-vinyl, industrial, beauty, tools).

## 6. What was checked before launch

An Opus critic and an Opus code reviewer found these problems. Each was fixed before the run:

- the runner crashed on start (a missing variable);
- averaging over n could hide a loss at small n, so each n is now judged alone;
- a skipped screening step looked like "everything was killed", so the next stage now refuses to start;
- the final table showed the wrong row for a hypothesis;
- a bootstrap over about 10 logs is too narrow, so a t bound is used instead;
- ties in the ranking were not paired between rules, so every rule now gets the same tie-break;
- the every-second-round rule could end up with fewer people than the baseline, so its budget now matches the baseline's realised people;
- the screening draws were reused later, so screening now has its own draws.

## 7. What this cannot show

- Amazon review logs dominate every split. "Confirmed" means confirmed on Amazon-dominated logs.
- Nine of the holdout logs were used in the earlier tracking study, where the baseline half-life was chosen. Only the 5 new categories are entirely new.
- Rounds here are long (about 9 months for Amazon). Short rounds are not tested.
- The reviewers are same-provider subagents, not an outside audit.

## 8. Files

- `configs/tourney_registry.json` (frozen), `scripts/ssp/tourney_core.py`, `run_tourney.py`, `an_tourney.py`, `run_tourney_night.sh`.
- Results: `results/small_sample_popularity/03_tables/tourney_*.csv` and `tourney_survivors.json`.
- Rerun: `nice -n 10 caffeinate -i sh scripts/ssp/run_tourney_night.sh` (resumable; two to three workers under the quiet governor).
