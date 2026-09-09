# Data management

## Decision

Use ordinary Git for the working dataset. Keep the original MovieLens archive
in the GitHub Release as frozen provenance, and commit a byte-preserving
separated CSV package under `data/raw/`.

The large CSV files are split into deterministic chunks below GitHub's 100 MiB
per-file limit. Git then provides ordinary branches, pull requests, diffs,
history, reverts, and blame for dataset changes.

## Current snapshot

1. Source: GroupLens MovieLens `ml-latest`.
2. Snapshot date: 20 July 2023.
3. Contents: 33,832,162 ratings and 2,328,315 tag applications across 86,537 movies from 330,975 users.
4. Release: [MovieLens snapshot](https://github.com/JerNB/Federated_Data_Gathering_Research/releases/tag/data-2023-07-20).
5. Archive size: 356,291,813 bytes.
6. Archive SHA 256: `21c09ce12e8062c6237011432fbd3acacb9e80d094f8400b1ce8c7592f725804`.
7. File checksums: `data/ml-latest.sha256`.

The release is immutable. A new upstream snapshot receives a new release,
manifest version, and Git package update.

## Package workflow

Fetch and verify the original source:

```sh
scripts/fetch_data.sh
scripts/fetch_data.sh --verify
```

Create or refresh the Git package:

```sh
python3 scripts/separate_dataset.py \
  --input ml-latest \
  --output-root data/raw \
  --source-manifest data/dataset_manifest.json \
  --chunk-manifest data/chunk_manifest.json \
  --force
```

Verify the package:

```sh
python3 scripts/separate_dataset.py --verify
```

The chunk manifest records the source manifest digest, source file digests,
per-chunk digests, and exact reassembled digests. `.gitattributes` prevents
Git from changing source line endings or attempting huge text diffs.

## Collaboration workflow

1. Pull the latest Git branch.
2. Prepare the complete input snapshot for the intended dataset version.
3. Run the documented separation command with `--force`.
4. Run `python3 scripts/separate_dataset.py --verify`.
5. Commit the generated data package and open a pull request.
6. Merge after review and validation.

`data/raw/` is the single Git-tracked data package. The frozen upstream
snapshot remains available from the dated release; Git history records every
package change, review, merge, and revert.

## Files in Git

1. `data/raw/` separated CSV files.
2. `data/chunk_manifest.json`.
3. `data/dataset_manifest.json`.
4. `data/ml-latest.sha256`.
5. Separation and validation scripts.
6. Experiment configurations and run records.

The ignored `ml-latest/` directory is only a local staging copy used to
rebuild or independently verify the tracked package.

## License

The MovieLens notice and usage conditions travel with every copy and
transformation. Full terms are in `ml-latest/README.txt`.

## Sources

1. [GitHub large file storage](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage)
2. [GitHub releases](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)
3. [MovieLens dataset paper](https://doi.org/10.1145/2827872)
4. [Experiment workflow](experiment_workflow.md)
