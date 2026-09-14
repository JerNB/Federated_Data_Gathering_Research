# Federated_Data_Gathering_Research

Research code over the MovieLens `ml-latest` dataset (snapshot generated
2023-07-20: 33,832,162 ratings and 2,328,315 tag applications across 86,537
movies from 330,975 users).

## Repository map

| Area | Path | Contents |
| --- | --- | --- |
| Proposal | `docs/research_proposal.md` | Canonical claim, estimands, controls, phases, evidence. |
| Index | `docs/README.md` | Reading order for every document. |
| Direction decision | `docs/direction_assessment.md` | Literature matrix, evidence cards, ranked decision, executed result. |
| Decision framework | `docs/exploration_goal.md` | How a data-gathering direction is promoted, deferred, or rejected. |
| Sources | `docs/literature_sources.md` | Primary-source register (S1–S29) with scope notes. |
| Sampling matrix | `docs/candidate_matrix.md` | Snapshot-calibration sampler catalog and boundaries. |
| Data and workflow | `docs/data_management.md`, `docs/experiment_workflow.md` | Package integrity and run procedure. |
| Later tracks | `docs/research_direction.md` | Deferred comparators and federated-system confirmation. |
| Experiment contracts | `configs/experiments/` | `sample_generalization_v1.json`, `fixed_cohort_budget_v1.json`. |
| Evidence | `results/explorations/` | Executed artifacts, reference artifacts, reports, figures. |

## Experiments

| Experiment | Question | Run | Validate |
| --- | --- | --- | --- |
| Snapshot sampling matrix | Which sampling mechanisms preserve a full-snapshot result? | `make sample-generalization` | `make validate-sample-generalization` |
| Fixed-cohort local-data budget | For the same users, how much local history does the recommendation result need? | `make fixed-cohort-budget` | `make validate-fixed-cohort-budget` |
| Everything tracked | — | — | `make validate-all` |

The repository contains a Git-tracked, byte-preserving CSV package under
`data/raw/`. The large source tables are separated into deterministic chunks
below GitHub's 100 MiB per-file limit. The tracked package is authoritative;
the original archive URL is recorded for provenance and any separately obtained
archive must pass the manifest checksums before use.

Fetch and verify the original source snapshot when a clean source copy is needed:

```sh
scripts/fetch_data.sh
scripts/fetch_data.sh --verify
```

Create or refresh the separated package with:

```sh
python3 scripts/separate_dataset.py \
  --input ml-latest \
  --output-root data/raw \
  --source-manifest data/dataset_manifest.json \
  --chunk-manifest data/chunk_manifest.json \
  --force
```

Verify every tracked chunk and its byte-for-byte reassembly with:

```sh
python3 scripts/separate_dataset.py --verify
```

The chunk manifest records the source checksum, every chunk checksum, the
reassembled checksum, and the source manifest checksum. Git tracks the
separated files and their history. No database conversion step is required.

Expected layout:

```text
data/raw/
  ratings/
    ratings-000.csv
    ratings-001.csv
  genome-scores/
    genome-scores-000.csv
    genome-scores-001.csv
  tags/
    tags-000.csv
    tags-001.csv
  movies.csv
  links.csv
  genome-tags.csv
  README.txt
```

## Full-data sampling-generalization matrix

The executed candidate matrix compares four sampling schemes and three fixed
recommendation models over seven fractions with ten draws below the full
fraction. It uses one frozen chronological split, panel, catalog, test target,
support rule, metric contract, and reference model. The candidate catalog,
primary/secondary/deferred labels, and interpretation boundary are in
`docs/candidate_matrix.md`.

Run the matrix:

```sh
make sample-generalization
```

Validate the tracked result contract:

```sh
make validate-sample-generalization
```

Current evidence is tracked under
`results/explorations/sample_generalization_full/`, including 732 draw rows,
84 aggregate rows, six figures, a reference artifact, and a concise report.
The dashboard exposes this run under the full-data candidate matrix panel.

## Fixed-cohort local-data-budget replay

The second experiment holds the client cohort fixed and varies only how much
collection-window local history each client contributes. It compares an equal
chronological cap with an item-tail reserve cap against all permitted history
from the same cohort, evaluated on later interactions from those same users.

Models are declared controls: deterministic popularity, rating-weighted
popularity, and item-item cosine, plus stochastic implicit ALS whose same-data
seed floor is published beside every effect. Item-item cosine is the primary
personalized probe because it has no random state.

```sh
make fixed-cohort-budget
make validate-fixed-cohort-budget
```

Evidence is tracked under `results/explorations/fixed_cohort_budget_v1/`.
The replay is an offline MovieLens study: it makes no federated-system,
availability, privacy, consent, or external-generalization claim.

## Interactive research dashboard

The repository includes a dependency-free local dashboard for inspecting the
tracked exploration metadata, split contract, support probes, run plans, and
exploration figures:

```sh
make dashboard
```

Open <http://127.0.0.1:8787>. The dashboard serves the API and static webpage
from the same local process. Use the snapshot selector to switch between
`raw_snapshot_preflight` and the explicit support probe. Override the bind
address or port when needed:

```sh
make dashboard DASHBOARD_HOST=127.0.0.1 DASHBOARD_PORT=8788
```

The app reads tracked metadata from `results/explorations/`, `results/run_records/`, `results/run_registry.jsonl`, and the editable board at `results/research_board.json` at request time. The full-data candidate matrix has a dedicated API panel with six tracked PNG figures and its report. Board edits are saved directly through the app; explicit allowlisted Run actions may launch background jobs and expose their status, while arbitrary shell commands remain rejected.

The research questions and proposal are stored canonically in `docs/research_proposal.md` and rendered in the dashboard's proposal panel. The candidate matrix and executed evidence are organized separately in `docs/candidate_matrix.md` and `results/explorations/sample_generalization_full/`; the older federated model direction remains a later confirmation track.

Each board row also stores an allowlisted command and editable parameters. The Run button can launch `make explore`, `make validate-experiment`, `make sample-generalization`, `make validate-sample-generalization`, or `create_run_record.py`; arbitrary shell commands are rejected. Mutations require a localhost origin and are persisted atomically.

## Dataset license and citation

The MovieLens license permits redistribution, including transformations, as long
as it is distributed under the same conditions; it also prohibits commercial use
without permission from GroupLens. Full terms are in `ml-latest/README.txt`
after fetching. Neither the University of Minnesota nor GroupLens endorses this
work.

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History
> and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4:
> 19:1–19:19. <https://doi.org/10.1145/2827872>
