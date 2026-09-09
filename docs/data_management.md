# Data management

## Decision

Keep code, scripts, manifests, and short documentation in Git. Keep the raw MovieLens snapshot outside the Git history. Use a versioned GitHub release asset for the pinned snapshot. `scripts/fetch_data.sh` downloads the archive into the ignored `ml-latest/` directory and checks every extracted file against `data/ml-latest.sha256`.

This gives the project a small clone, one repeatable setup command, and a clear data version. It also avoids the 100 MiB limit for ordinary GitHub files.

## Current snapshot

1. Source: GroupLens MovieLens `ml-latest`.
2. Snapshot date: 20 July 2023.
3. Contents: 33,832,162 ratings and 2,328,315 tag applications across 86,537 movies from 330,975 users.
4. Release: [MovieLens snapshot](https://github.com/JerNB/Federated_Data_Gathering_Research/releases/tag/data-2023-07-20).
5. Archive size: 356,291,813 bytes.
6. Archive SHA 256: `21c09ce12e8062c6237011432fbd3acacb9e80d094f8400b1ce8c7592f725804`.
7. File checksums: `data/ml-latest.sha256`.

The release asset is below GitHub's 2 GiB release asset limit. Treat a published data asset as fixed after release. Create a new dated release for a new snapshot.

## Collaborator workflow

```sh
scripts/fetch_data.sh
scripts/fetch_data.sh --verify
```

The first command downloads and verifies the data. The second command verifies an existing copy without downloading again.

The following files belong in Git:

1. `scripts/fetch_data.sh`.
2. `data/ml-latest.sha256`.
3. Documentation that records the source, license, snapshot date, and retrieval steps.
4. Code that creates derived data from the source snapshot.

The following files stay outside Git:

1. `ml-latest/`.
2. `ml-latest.zip`.
3. Large derived files and temporary downloads.
4. Credentials for any storage service.

## What other projects use

1. **GitHub Releases** store large binary assets beside a tagged repository version. This is simple for a fixed public snapshot and works well with a download script.
2. **Git LFS** stores a small pointer in Git and the large object in LFS storage. It supports ordinary Git collaboration. Storage and download quotas add cost and setup, so it fits active file editing better than a fixed source snapshot.
3. **DVC** stores lightweight data pointers in Git and puts the data in a shared remote such as object storage. It fits derived datasets, experiment outputs, and teams that need branchable data versions. It adds a tool and a remote configuration step.
4. **Hugging Face dataset repositories** provide public hosting, version history, commit diffs, metadata, dataset cards, and a dataset viewer. They fit public discovery, interactive use, and collaborative dataset updates.
5. **Zenodo records** provide long term preservation and a DOI. Published files stay fixed, and each update receives a new version. This fits a paper or final public release rather than daily editing.
6. **Recommendation projects** such as Microsoft Recommenders keep code and examples in Git, then prepare or download datasets through reusable data utilities. This keeps repository history focused on code and makes data setup explicit.

## Finalized path

Use separate roles for source preservation, active data work, and experiment recording.

1. Keep the current GitHub Release as the frozen provenance archive for the MovieLens source snapshot.
2. Prepare one versioned Kaggle Dataset as the public working registry. It will contain the raw archive, the canonical SQLite package, a small preview file, metadata, checksums, and a data card.
3. Build the canonical local package with `scripts/build_dataset_sqlite.py`. The raw files remain unchanged and the generated package stays outside Git.
4. Keep the manifest, transformation configuration, objective formulas, experiment configurations, run schema, and milestone reports in GitHub.
5. Use one shared Weights and Biases project for live runs, metrics, charts, and artifact lineage. The repository run schema remains the stable interface for the tracker.
6. Create a Zenodo record after a paper dataset or benchmark release is stable and needs a DOI.

The Kaggle Dataset is a planned working registry. No external data upload is part of this repository change. Before publication, carry the MovieLens notice and usage conditions into the data card and verify the selected registry settings.

Git LFS, DVC, and Hugging Face remain valid alternatives. Git LFS suits active file versioning, DVC suits shared object storage, and Hugging Face suits a code first dataset hub. Each adds a separate service or workflow, so none is the current primary registry.

## New data release checklist

1. Record the upstream source, license, snapshot date, and transformation steps.
2. Keep the original directory layout when practical.
3. Create a dated release and upload the archive as an asset.
4. Record the archive SHA 256 and per file SHA 256 values.
5. Update the fetch script, manifest, README, and this document.
6. Run a fresh download and a separate verification pass.
7. Link the release from any public dataset page or paper.

## Sources

1. [GitHub large file storage](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage)
2. [GitHub releases](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)
3. [DVC remote storage](https://dvc.org/doc/user-guide/data-management/remote-storage)
4. [Hugging Face dataset sharing](https://huggingface.co/docs/datasets/en/share)
5. [Zenodo records](https://help.zenodo.org/docs/deposit/about-records/)
6. [Microsoft Recommenders](https://github.com/recommenders-team/recommenders)
7. [Conversation archive for this project](https://chatgpt.com/s/cx_6a9f72cef5ac8191b31ec6c5c5bbf5ee)
8. [Kaggle datasets](https://www.kaggle.com/docs/datasets)
9. [Weights and Biases artifacts](https://docs.wandb.ai/models/artifacts)
10. [Experiment workflow](experiment_workflow.md)
