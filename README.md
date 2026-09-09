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

data/editorial/
  movies.csv
  links.csv
```

## Dataset license and citation

The MovieLens license permits redistribution, including transformations, as long
as it is distributed under the same conditions; it also prohibits commercial use
without permission from GroupLens. Full terms are in `ml-latest/README.txt`
after fetching. Neither the University of Minnesota nor GroupLens endorses this
work.

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History
> and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4:
> 19:1–19:19. <https://doi.org/10.1145/2827872>
