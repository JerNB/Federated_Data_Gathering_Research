# Results registry

`run_registry.jsonl` is the append only summary of every planned, running, completed, or failed experiment record.

Full records are written to `results/run_records/` when the first record is created. Lifecycle updates replace that full record while preserving `created_at_utc` and adding `updated_at_utc`. Large checkpoints, data packages, and tracker artifacts stay in their external artifact store.

## Sampling-generalization evidence

The tracked exploratory result
`explorations/sample_generalization_full/` contains the machine-readable
candidate summary, frozen-reference metadata, six figures, and Markdown report
for the full pinned MovieLens matrix. It contains 732 draw rows and 84
aggregate rows across four sampling schemes, three fixed models, seven
fractions, and ten sub-full replicates.

Validate the result contract with:

```sh
make validate-sample-generalization
```

The result is an in-reference approximation study. The report explicitly
separates fixed-reference metrics from sample-native exclusion sensitivity and
does not claim independent external generalization.

## Fixed-cohort local-data-budget evidence

`explorations/fixed_cohort_budget_v1/` holds the second experiment: one fixed
2,000-user cohort, global chronological pilot/collection/future windows, two
retention policies, six per-client caps, and four declared models, for 48
policy/model rows plus a frozen reference artifact and report.

Validate the result contract with:

```sh
make validate-fixed-cohort-budget
```

The reference artifact publishes the deterministic item-item probe metadata and
the implicit-ALS same-data seed-variance floor. A model effect smaller than that
floor is not evidence about the data budget.

Create a planned record with:

```sh
scripts/create_run_record.py --variant global_mf --run-id milestone_1_global_plan
```

Each registry line records the dataset version, code commit, worktree state, model variant, status, metrics, and full record path. Lifecycle updates append another summary line for the same run identifier, preserving the history of state changes.
