# Dataset package

## Source snapshot

The source snapshot is MovieLens `ml-latest`, generated on 20 July 2023.
`data/dataset_manifest.json` and `data/ml-latest.sha256` pin the upstream
archive and every original file checksum.

```sh
scripts/fetch_data.sh
scripts/fetch_data.sh --verify
```

The download creates the ignored `ml-latest/` staging directory. The original
archive remains frozen in the GitHub Release.

## Git-tracked package

The repository stores a byte-preserving separated copy under `data/raw/`.
Large CSV files are split into stable chunks below the GitHub 100 MiB
per-file limit. Small files remain single CSV files.

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

The chunk manifest is:

```text
data/chunk_manifest.json
```

It records the source manifest checksum, source file checksums, chunk paths,
per-chunk checksums, and byte-for-byte reassembled checksums.

## Create and verify the package

```sh
python3 scripts/separate_dataset.py \
  --input ml-latest \
  --output-root data/raw \
  --source-manifest data/dataset_manifest.json \
  --chunk-manifest data/chunk_manifest.json \
  --force

python3 scripts/separate_dataset.py --verify
```

The separation script preserves the source bytes, including line endings.
The first chunk contains the source header. Following chunks contain
continuation rows, so the chunks reassemble exactly to the original file.

## Collaboration

`data/raw/` is the single Git-tracked package. To publish a changed dataset,
prepare its complete input snapshot, regenerate the package and manifest with
`scripts/separate_dataset.py --force`, and commit the resulting files. Do not
normalize line endings or edit chunk boundaries by hand; Git preserves package
history, reviews, merges, and reverts.

## Data rules


1. The original source snapshot is immutable.
2. Separated package files preserve source bytes.
3. Every package refresh must pass the chunk manifest verification.
4. Train, validation, and test construction uses training-only information.
5. Dataset commits and source versions are pinned in experiment records.
6. The MovieLens notice and usage conditions travel with the package.
7. No credentials or private storage settings enter this repository.
