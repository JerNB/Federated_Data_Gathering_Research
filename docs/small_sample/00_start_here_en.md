# Start here: Small-Sample Popularity Estimate

Chinese version: [`00_start_here_zh.md`](00_start_here_zh.md). Written for colleagues who want to read and judge the work.

## 1. What this is, in three sentences

Take a small sample of people from a larger community, keep a few of each person's past choices, and use only that
to build a ranked list of items. **How good is that list, how many people do you need, and when is a personalized
recommender trained on the same sample worth more than just showing the popularity list?**
We treat the popularity list like a statistic (a mean, a variance), except the thing estimated is a ranking.

<!-- HEADLINE -->

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
