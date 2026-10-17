# Start here: Small-Sample Popularity Estimate

Chinese version: [`00_start_here_zh.md`](00_start_here_zh.md). Written for colleagues who want to read and judge the work.

## 1. What this is, in three sentences

Take a small sample of people from a larger community, keep a few of each person's past choices, and use only that
to build a ranked list of items. **How good is that list, how many people do you need, and when is a personalized
recommender trained on the same sample worth more than just showing the popularity list?**
We treat the popularity list like a statistic (a mean, a variance), except the thing estimated is a ranking.

## Headline

- A popularity list from a few hundred people is already close to the full-community list in soft quality. **The bigger
  gap is time, not sample size:** the community's earlier list is far from its later list in all 10 datasets we could
  judge, a recent-window sample predicts the later list better than a uniform sample in 10 of 11, and recency-weighted
  counts beat plain counts in 80% of cells. The repair is small (about 2% of the gap on the list), so drift is the
  open problem.
- A personalized model trained on the same sample is still improving at the largest tested size, while the popularity
  list saturates much earlier; in a post-hoc small-N stage it beats popularity from as few as 20 people in 3 of 7
  datasets.
- The three pre-registered predictions did **not** hold as written (C1 inconclusive, C2 falsified, C3 mixed); the reasons
  and our checks of our own explanations are in [`02_results_en.md`](02_results_en.md) section 2.

## 2. Read in this order

| # | File | What you get | Time |
| ---: | --- | --- | --- |
| 1 | this page | the question, the headline, the file map | 5 min |
| 2 | [`01_preregistration_en.md`](01_preregistration_en.md) | what we predicted and how each claim would be judged, **written before the runs** (plus dated amendments) | 15 min |
| 3 | [`02_results_en.md`](02_results_en.md) | every result: claim, verdict, plain reading, figure | 20 min |
| 4 | `results/small_sample_popularity/03_tables/cells_*.csv` | the numbers behind every figure: one row per cell, mean and standard error | as needed |

## 3. Where things live (numbered, in the order they are produced)

```text
docs/small_sample/                      the three documents above, English and Chinese
datasets/fetch.sh                       downloads the public sources for 11 datasets (raw files are not in Git)
scripts/ssp/prep_data.py                raw files -> one compact format per dataset
scripts/ssp/frame.py, est.py, ...       the engine: sampling, estimators, metrics
scripts/ssp/run_all.py                  the task queue (resumable); run_overnight.sh runs it quietly
scripts/ssp/an_*.py                     one analysis script per family of claims
results/small_sample_popularity/
  00_safety/    fan test (calibration.csv), the governor's log, the queue log
  01_datasets/  datasets.json: size, concentration and theory numbers per dataset
  02_runs/      raw rows, one per random draw (large, not in Git)
  03_tables/    cell means (cells_*.csv), claim tables, *_verdicts.json
  04_figures/   every figure used in the results document
```

## 4. How to rerun

```sh
make venv                               # pinned Python and packages (once)
datasets/fetch.sh                       # download the public data
.venv/bin/python3 scripts/ssp/prep_data.py --all
.venv/bin/python3 scripts/ssp/describe_datasets.py
scripts/ssp/run_overnight.sh            # all stages; safe to stop and restart
.venv/bin/python3 scripts/ssp/aggregate.py
# analysis, one script per family (each writes tables and figures):
for s in an_confirm2 an_window an_explore_a an_explore_b an_extra an_stoprule an_curves an_hunt; do .venv/bin/python3 scripts/ssp/$s.py; done
.venv/bin/python3 scripts/ssp/make_report.py
```

## 5. The run is built to be quiet

It was run overnight on a laptop with the fans kept at idle. `scripts/ssp/quiet.py` reads die temperature, fan speed,
memory and disk every 3 seconds (`macmon`, no sudo) and slows or freezes the workers before the fans rise. Details and
the fan measurements are in `results/small_sample_popularity/00_safety/`.

## 6. What this does not show

- Samples are random draws from a community, not a real small group (except the subgroup stage).
- The data are public volunteer or platform logs; trends are "in these data".
- Federation is simulated; there are no devices, dropouts or real secure aggregation.
- Personalized models were trained on at most 1,500 candidate items to keep the machine quiet; see amendment A15.

## 7. Later work: the weekly-memory programme (docs 03–17 and 90)

The main study (docs 01–02) asked how many people are needed. Everything after it asks a server's practical question: **how long should its memory be, and can it choose that memory from its own data?** Every later document is numbered in the order it was written; "pre-registered" means the plan was committed before the run, "post hoc" means it was looked at after.

| Doc | English | What you get |
| ---: | --- | --- |
| 03 | [`03_conclusions_en.md`](03_conclusions_en.md) | the main study in plain words: eight terms, the three predictions and what happened |
| 04 | [`04_next_hypotheses_en.md`](04_next_hypotheses_en.md) | candidate next ideas at the end of the main study |
| 05 | [`05_refresh_and_two_round_en.md`](05_refresh_and_two_round_en.md) | seasonality and two-round asking: neither idea helped |
| 06–07 | [`06_tracking_design_en.md`](06_tracking_design_en.md), [`07_tracking_results_en.md`](07_tracking_results_en.md) | tracking popularity round by round on 22 time-stamped logs: 100 people per round give a list about as good as a full count 1.3 to 8 rounds old |
| 08–09 | [`08_tournament_design_en.md`](08_tournament_design_en.md), [`09_tournament_results_en.md`](09_tournament_results_en.md) | 12 candidate ideas screened with a fixed error budget |
| 10–12 | [`10_deep_plan_en.md`](10_deep_plan_en.md), [`11_deep_cards_en.md`](11_deep_cards_en.md), [`12_deep_results_en.md`](12_deep_results_en.md) | 30 pre-registered questions on 30 logs: no alternative to the simple decayed sum gained; a server can score its own lists without ground truth; privacy costs a lot |
| 13 | [`13_weekly_results_en.md`](13_weekly_results_en.md) | weekly rounds: the best memory is much longer and depends on the number of people per round (about 32, 16, 8, 4, 2 rounds at 30, 100, 300, 1,000, 3,000 people) |
| 14 | [`14_selfsel_exploration_en.md`](14_selfsel_exploration_en.md) | post hoc: a server that publishes the memory whose recent self-score is best loses little against the best fixed memory |
| 15 | [`15_selfsel_confirmation_en.md`](15_selfsel_confirmation_en.md) | pre-registered confirmation of doc 14 on a second window: confirmed, mainly from about 100 people per round |
| 16 | [`16_selfsel_stress_en.md`](16_selfsel_stress_en.md) | pre-registered stress tests: volunteer bias, privacy noise, churn of the published list |
| 17 | [`17_selfsel_guard_en.md`](17_selfsel_guard_en.md) | pre-registered test on a third window: a selector seeded with prior self-scores fixes the weak regions (30 people; monthly rounds) |
| 90 | [`90_materials_ledger_en.md`](90_materials_ledger_en.md) | every dataset (with hashes), tool and literature source used, and the safety rules |

Numbers and decision rules for docs 13–17 are in `results/small_sample_popularity/03_tables/{weekly,confirm,stress,guard}/` and in the registries `configs/*_registry.json`. All reviews of these documents by subagents are same-provider checks, not an outside audit.
