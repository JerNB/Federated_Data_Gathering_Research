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

## Recommended path

Use a layered workflow.

1. Keep the current GitHub release workflow for the pinned MovieLens source snapshot. It is already implemented and is the simplest path for this repository.
2. Add a Hugging Face dataset mirror when the project needs a public data page, browser access, streaming, or visible dataset collaboration. Give the mirror a dataset card with the MovieLens license, source, snapshot date, checksum, and limitations. Review redistribution terms before uploading.
3. Add DVC when the project creates derived partitions, processed files, or experiment artifacts that several collaborators need to version together. Store DVC data in a shared remote and keep credentials local.
4. Create a Zenodo record for a stable paper dataset or benchmark release when a DOI and long term archive are useful.
5. Keep GitHub as the home for code, experiment definitions, documentation, and links to each data version.

This avoids moving the current source data into Git LFS or DVC before the project needs their extra workflow. It also leaves a clear route to public presentation and active data collaboration.

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
