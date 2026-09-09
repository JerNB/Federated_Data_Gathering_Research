# Dataset package

## Source snapshot

The source snapshot is MovieLens `ml-latest`, generated on 20 July 2023. Git tracks its manifest and checksums. The raw files are fetched into the ignored `ml-latest/` directory.

```sh
scripts/fetch_data.sh
scripts/fetch_data.sh --verify
```

The complete machine readable description is in `data/dataset_manifest.json`.
The canonical SQLite schema is in `data/canonical_schema.json`.

## Planned public package layout

The local SQLite file is the working package. The target public registry is one versioned Kaggle Dataset. Each version will contain a data card, the source notice, checksums, and a reproducible transformation record.

```text
raw/
  ml-latest.zip

canonical/
  movielens.sqlite

preview/
  ratings_sample.csv

metadata/
  dataset.yaml
  schema.yaml
  checksums.sha256
```

The raw archive preserves the upstream snapshot. The SQLite file provides indexed tables for model preparation and public exploration. The preview file supports quick inspection. A later format change creates a new dataset version.

## Build the local working package

```sh
scripts/build_dataset_sqlite.py \
  --input ml-latest \
  --output data_working/movielens.sqlite \
  --manifest data/dataset_manifest.json
```

The generated package is a derived artifact. It stays outside Git and records the source manifest digest, source version, build time, selected tables, and row counts in its `metadata` table.

Use `--include-genome` when genome tables are required. Use `--limit` only for a smoke build. A limited build is never a research dataset.

## Data rules

1. Raw files are read only.
2. Every transformation has a versioned configuration and a new output manifest.
3. Train, validation, and test construction uses training only information for routing and catalog selection.
4. Dataset versions are pinned in every experiment record.
5. The MovieLens notice and usage conditions travel with every public transformation.
6. No credentials or private storage settings enter this repository.
