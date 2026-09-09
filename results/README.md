# Results registry

`run_registry.jsonl` is the append only summary of every planned, running, completed, or failed experiment record.

Full records are written to `results/run_records/` when the first record is created. Large checkpoints, data packages, and tracker artifacts stay in their external artifact store.

Create a planned record with:

```sh
scripts/create_run_record.py --variant global_mf --run-id milestone_1_global_plan
```

Each registry line records the dataset version, code commit, worktree state, model variant, status, metrics, and full record path. Reusing a run identifier with `--force` appends a new lifecycle line and replaces the full record at that path.
