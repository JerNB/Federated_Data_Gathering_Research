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
  objectives/
  experiments/

data/
  dataset_manifest.json
  ml-latest.sha256
  chunk_manifest.json
  raw/

experiments/
  run_record.schema.json

results/
  README.md
  run_registry.jsonl
```

Created on first use or during later milestones:

```text
results/
  exported_metrics.csv
  run_records/

reports/
  milestone_1/
```

The separated data package is tracked in Git. Checkpoints and temporary model outputs remain outside Git in `runs/` and `artifacts/`.
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

Every run records:

1. A stable run identifier.
2. The experiment identifier.
3. The code commit.
4. Whether the worktree was clean when the record was created.
5. The worktree status when it was not clean.
6. The dataset identifier and source version.
7. The source manifest SHA 256 digest.
8. The Git package chunk manifest SHA 256 digest.
9. The configuration path and digest.
10. The model variant.
11. The objective and score formula.
12. The split method and seed.
13. The selected user count and catalog counts.
14. The support report for every oracle cluster.
15. Parameters, metrics, artifacts, and notes.

The full record is stored in `results/run_records/`. Its compact summary is appended to the tracked `results/run_registry.jsonl` file. The schema is in `experiments/run_record.schema.json`.


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

## First comparison

The first experiment contains exactly two model variants:

1. Global matrix factorization with local user factors and one shared item factor matrix.
2. Capacity matched oracle clustered matrix factorization with local user factors and one item factor matrix per declared cluster.

The pilot uses a fixed deterministic user sample stratified across oracle clusters. Each cluster receives its configured support filtered catalog. Both model variants use the same catalog within each cluster. The all cluster intersection is retained as a secondary catalog for reporting.

The support report includes item support statistics for every cluster, each cluster catalog size, and the all cluster intersection size. If any cluster fails its support condition, increase the user population or reduce the cluster count before interpreting a model difference. Full population and full catalog confirmation happens after the pilot protocol and hyperparameters are frozen.

Routing methods are later model variants. They do not enter the first comparison.

The support selection tool is `scripts/select_pilot_catalog.py`. It reads a training only interaction file and oracle cluster map, selects equal user counts per cluster, filters the catalog separately for each cluster, and writes the per cluster support report.

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
