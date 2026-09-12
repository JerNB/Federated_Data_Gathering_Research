# Federated_Data_Gathering_Research

Research code over the MovieLens `ml-latest` dataset (snapshot generated
2023-07-20: 33,832,162 ratings and 2,328,315 tag applications across 86,537
movies from 330,975 users).

## Project documents

1. [Data management](docs/data_management.md)
2. [Experiment workflow](docs/experiment_workflow.md)
3. [Research direction](docs/research_direction.md)
4. [Dataset manifest](data/dataset_manifest.json)
5. [Chunk manifest](data/chunk_manifest.json)

## Data setup

The repository contains a Git-tracked, byte-preserving CSV package under
`data/raw/`. The large source tables are separated into deterministic chunks
below GitHub's 100 MiB per-file limit. The original archive remains available
from the `data-2023-07-20` release for provenance and independent verification.

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

The app reads tracked metadata from `results/explorations/`, `results/run_records/`, `results/run_registry.jsonl`, and the editable board at `results/research_board.json` at request time. Charts render from the JSON numbers. PNG figures are optional local enrichment; if `artifacts/` has not been generated, the dashboard shows a clear placeholder instead of a broken image. Board edits are saved directly through the app; explicit allowlisted Run actions may launch background jobs and expose their status, while arbitrary shell commands remain rejected.

The research questions and proposal are stored canonically in `docs/research_proposal.md` and rendered in the dashboard's proposal panel. The document keeps literature review before benchmark freezing, treats small-to-large generalization as the primary scientific objective, and leaves the scaling unit and federated definition open until the review narrows them.

Each board row also stores an allowlisted command and editable parameters. The Run button can launch `make explore`, `make validate-experiment`, or `create_run_record.py` with those parameters; arbitrary shell commands are rejected. Mutations require a localhost origin and are persisted atomically.

## Dataset license and citation

The MovieLens license permits redistribution, including transformations, as long
as it is distributed under the same conditions; it also prohibits commercial use
without permission from GroupLens. Full terms are in `ml-latest/README.txt`
after fetching. Neither the University of Minnesota nor GroupLens endorses this
work.

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History
> and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4:
> 19:1–19:19. <https://doi.org/10.1145/2827872>
