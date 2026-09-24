# Experiment workflow

## Purpose

The repository stores the research contract. A shared experiment tracker stores live runs and artifacts. A report connects the two.

The workflow has four layers:

1. **Source data**: the pinned MovieLens snapshot and its manifest.
2. **Git data package**: byte-preserving separated CSV files and their chunk manifest.
3. **Experiment configuration**: the formula, split, model variant, capacity budget, and evaluation rules.
4. **Run record**: the Git commit, data package manifest, support report, metrics, and artifacts.

A command can execute a run. The configuration and run record explain what the command means.

## Repository layout

Tracked now:

```text
configs/
  objectives/                objective files
  experiments/               primary and deferred experiment contracts
  run_record.schema.json     run-record schema

data/
  dataset_manifest.json      source snapshot identity
  chunk_manifest.json        Git package identity
  raw/                       separated CSV package

docs/
  README.md                  documentation index and reading order
  data_management.md
  experiment_workflow.md
  literature_sources.md      cited-source register
  research_direction.md      deferred candidate tracks
  research_proposal.md       canonical primary proposal
  exploration_goal.md        direction promotion framework
  direction_assessment.md    executed direction decision
  candidate_matrix.md        sampler catalog

results/
  explorations/              tracked preflight and full candidate outputs
  run_records/               planned/completed run records
  research_board.json        editable research tasks
  run_registry.jsonl         compact run index
  README.md

scripts/
  run_sample_generalization.py
  validate_sample_generalization.py
  run_fixed_cohort_budget.py
  validate_fixed_cohort_budget.py
  create_run_record.py
  explore_dataset.py
  select_pilot_catalog.py
  validate_experiment.py
  selftest.py
  dashboard_server.py

web/
  index.html
  app.js
  styles.css

Makefile
requirements-exploration.txt
```

Generated local outputs are intentionally not tracked:

```text
artifacts/<run_id>/            exploration figures and dashboard job logs
ml-latest/                     full upstream source snapshot
```

The separated data package is tracked in Git. Checkpoints and temporary model
outputs remain outside Git. Tracked exploration metadata, run records, the
proposal, and the research board are part of the reproducible project record.

## Reproducible exploration

The raw-data visualization pass is implemented in `scripts/explore_dataset.py`
and exposed through the repository `Makefile`:

```sh
make explore RUN_ID=raw_snapshot_preflight
```
`make explore` reads the tracked `data/raw/` package. It does not require the
ignored `ml-latest/` staging directory. Use `make verify-source` separately
when the full upstream staging snapshot also needs checksum verification.

The command validates the pinned source package first, then writes the tracked
metadata file `results/explorations/<run_id>.json` and local figures under
`artifacts/<run_id>/figures`. The metadata records the script hash,
configuration and manifest hashes, chunk headers, split semantics, seeded
timestamp tie handling, sampling seed, rank-ordered heatmap strata, optional
cluster-map hash, and the configured per-cluster-equivalent global support
probe. Use `CLUSTER_MAP=path/to/user_clusters.csv` to add cluster-conditioned
descriptives without changing the exploration code.

Figure artifacts are local and ignored; run metadata under
`results/explorations/`, the script, and the command contract are tracked.
Exploration is not a model run and therefore does not create a completed
`results/run_records/` entry: the model-run lifecycle requires a training-only
support report and a clean worktree.

## Run identity

Every run record must identify:

1. A stable run identifier.
2. The experiment and sampling scheme.
3. The Git commit and worktree state.
4. The dataset version and manifest digests.
5. The model, objective, split, sample fraction, replicate, and seed.
6. Realized users, items, interactions, support, and evaluation-panel counts.
7. Metrics, uncertainty, figures, interpretation, and artifact paths.

The full record is stored in `results/run_records/`. The compact summary is
appended to `results/run_registry.jsonl`. The primary candidate matrix is
stored in `results/sample_generalization/candidate_summary.json`.



## Shared tracking

The initial shared tracking target is one public Weights and Biases project. Each run records configuration values, metrics, charts, tables, and output artifacts. Dataset packages and model outputs receive explicit artifact names and versions.

The repository remains independent of the tracker. A self hosted MLflow service can replace the hosted tracker later because both systems can consume the same run record fields.

## Result lifecycle

1. Commit an experiment configuration before running it.
2. Create a planned run record from that configuration.
3. Verify the pinned Git data package and record its source and chunk manifest digests.
4. Log the formula, parameters, metrics, and artifacts during execution.
5. Record the support report before interpreting ranking results.
6. Export a compact result row and report after the run finishes.
7. Link the public report to the data package commit, dataset version, and tracker run.

A completed result without its dataset version, formula, split, support report, or code commit is incomplete.

Create a planned record with:

```sh
scripts/create_run_record.py --variant global_mf --run-id milestone_1_global_plan
```

The command writes one full record and appends one summary line. Lifecycle updates require `--force` and must follow `planned` to `running` to `completed` or `failed`. Updates preserve `created_at_utc` and add `updated_at_utc`. Completed records require a clean worktree, metrics, and a passed support report.

Run the standard-library contract self-test with:

```sh
python3 scripts/selftest.py
```

It exercises selector composition, schema rejection, identity pinning, and lifecycle transitions.

## Full-data sampling-generalization run

The primary experiment is `configs/experiments/sample_generalization_v1.json`.
Its candidate catalog and status labels are documented in
`docs/candidate_matrix.md`. The reproducible runner is
`scripts/run_sample_generalization.py`.

The executed result is tracked under
`results/explorations/sample_generalization_full/`:

```text
results/explorations/sample_generalization_full/
  candidate_summary.json
  reference_artifact.json
  report.md
  figures/*.png
```

The full matrix uses one frozen full-data reference, the same panel, catalog,
split, metrics, exclusions, and seeds for every sample cell:

```sh
python3 scripts/run_sample_generalization.py \
  --panel-size 2000 \
  --fractions 0.01,0.025,0.05,0.1,0.25,0.5,1.0 \
  --replicates 10 \
  --models popularity,rating_weighted_popularity,item_item_cosine \
  --support-threshold 20 \
  --item-item-top-k 100 \
  --output results/explorations/sample_generalization_full
```

Validate the stored contract and artifact tree with:

```sh
make validate-sample-generalization
```

The result is an in-reference approximation study. It does not claim
independent external generalization. Deferred model and sampling candidates are
listed explicitly rather than being presented as executed evidence.

## Public report contents

Each milestone report should include:

1. Research question.
2. Dataset version and license notice.
3. Exact formulas.
4. Model ownership.
5. Data split and support rules.
6. Capacity and compute controls.
7. Result tables with uncertainty intervals.
8. Per cluster support statistics.
9. Interpretation and limitations.
10. Links to configuration, code commit, dataset version, and run records.

## Sources

1. [Weights and Biases experiment tracking](https://docs.wandb.ai/models/track)
2. [Weights and Biases artifacts](https://docs.wandb.ai/models/artifacts)
3. [MLflow tracking](https://mlflow.org/docs/latest/ml/tracking/)
4. [Kaggle datasets](https://www.kaggle.com/docs/datasets)
