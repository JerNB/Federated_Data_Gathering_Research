# Results registry

`run_registry.jsonl` is the append only summary of every planned, running, completed, or failed experiment record.

Full records are written to `results/run_records/` when the first record is created. Lifecycle updates replace that full record while preserving `created_at_utc` and adding `updated_at_utc`. Large checkpoints, data packages, and tracker artifacts stay in their external artifact store.

Create a planned record with:

```sh
scripts/create_run_record.py --variant global_mf --run-id milestone_1_global_plan
```

Each registry line records the dataset version, code commit, worktree state, model variant, status, metrics, and full record path. Lifecycle updates append another summary line for the same run identifier, preserving the history of state changes.
