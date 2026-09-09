# Experiment workflow

## Purpose

The repository stores the research contract. A shared experiment tracker stores live runs and artifacts. A report connects the two.

The workflow has four layers:

1. **Source data**: the pinned MovieLens snapshot and its manifest.
2. **Data package**: a versioned canonical package derived from the source snapshot.
3. **Experiment configuration**: the formula, split, model variant, capacity budget, and evaluation rules.
4. **Run record**: the code commit, dataset version, support report, metrics, and artifacts.

A command can execute a run. The configuration and run record explain what the command means.

## Repository layout

```text
configs/
  objectives/
  experiments/

data/
  dataset_manifest.json
  ml-latest.sha256

experiments/
  run_record.schema.json

reports/
  milestone_1/

results/
  exported_metrics.csv
  run_registry.jsonl
```

Large data packages, checkpoints, and temporary files remain outside Git. The current ignore rules reserve `data_working/`, `runs/`, and `artifacts/` for these outputs.

## Run identity

Every run records:

1. A stable run identifier.
2. The experiment identifier.
3. The code commit.
4. The dataset identifier and version.
5. The SHA 256 digest of the dataset manifest.
6. The configuration path and digest.
7. The model variant.
8. The objective and score formula.
9. The split method and seed.
10. The selected user count and catalog count.
11. The support report for every oracle cluster.
12. Parameters, metrics, artifacts, and notes.

The schema is in `experiments/run_record.schema.json`.

## Shared tracking

The initial shared tracking target is one public Weights and Biases project. Each run records configuration values, metrics, charts, tables, and output artifacts. Dataset packages and model outputs receive explicit artifact names and versions.

The repository remains independent of the tracker. A self hosted MLflow service can replace the hosted tracker later because both systems can consume the same run record fields.

## Result lifecycle

1. Commit an experiment configuration before running it.
2. Create a planned run record from that configuration.
3. Load the pinned dataset package and record its manifest digest.
4. Log the formula, parameters, metrics, and artifacts during execution.
5. Record the support report before interpreting ranking results.
6. Export a compact result row and report after the run finishes.
7. Link the public report to the dataset version, code commit, and tracker run.

A completed result without its dataset version, formula, split, support report, or code commit is incomplete.

## First comparison

The first experiment contains exactly two model variants:

1. Global matrix factorization with local user factors and one shared item factor matrix.
2. Capacity matched oracle clustered matrix factorization with local user factors and one item factor matrix per declared cluster.

The pilot uses a fixed deterministic user sample. Its catalog is restricted using training only support. An item enters the pilot catalog only when every oracle cluster has at least the configured minimum number of training interactions for that item.

The support report includes item support statistics for every cluster. If the support condition fails, increase the user population or reduce the cluster count before interpreting a model difference. Full population and full catalog confirmation happens after the pilot protocol and hyperparameters are frozen.

Routing methods are later model variants. They do not enter the first comparison.

The support selection tool is `scripts/select_pilot_catalog.py`. It reads a training only interaction file and oracle cluster map, selects the fixed user population, filters the catalog, and writes the per cluster support report. A failed support check stops interpretation of the comparison.

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
