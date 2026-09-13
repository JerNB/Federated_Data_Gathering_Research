#!/usr/bin/env bash
#
# Fetch and verify the MovieLens `ml-latest` snapshot this project is pinned to
# (generated 2023-07-20). The raw data is not stored in git: `ratings.csv` and
# `genome-scores.csv` exceed GitHub's 100 MiB per-file limit.
#
# Usage:
#   scripts/fetch_data.sh            # fetch into ./ml-latest if missing, then verify
#   scripts/fetch_data.sh --verify   # verify an existing ./ml-latest, no download
#   scripts/fetch_data.sh --force    # re-download even if ./ml-latest exists
#
# Override the source with DATA_URL (accepts any curl-supported URL, incl. file://).

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# The release URL is a pinned provenance source when its asset is available.
# The tracked data/raw package is sufficient for experiments; any alternate
# source must pass the checksum manifest below.
DATA_URL="${DATA_URL:-https://github.com/JerNB/Federated_Data_Gathering_Research/releases/download/data-2023-07-20/ml-latest.zip}"
UPSTREAM_URL="https://files.grouplens.org/datasets/movielens/ml-latest.zip"

checksums="data/ml-latest.sha256"
dest="ml-latest"
mode="fetch"

case "${1-}" in
  --verify) mode="verify" ;;
  --force)  mode="force" ;;
  "")       ;;
  *) echo "unknown argument: $1" >&2; exit 2 ;;
esac

die() { echo "error: $*" >&2; exit 1; }

# `sha256sum` on GNU/Linux, `shasum -a 256` on macOS.
check_sums() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum --quiet -c "$checksums"
  else
    shasum -a 256 -c "$checksums" >/dev/null
  fi
}

extract() {
  local zip="$1"
  if command -v unzip >/dev/null 2>&1; then
    unzip -q -o "$zip" -d .
  else
    tar -xf "$zip" -C .   # bsdtar reads zip archives
  fi
}

[[ -f "$checksums" ]] || die "missing checksum manifest: $checksums"

if [[ "$mode" == "verify" ]]; then
  [[ -d "$dest" ]] || die "$dest/ not found; run scripts/fetch_data.sh first"
  check_sums || die "checksum mismatch: $dest does not match the pinned snapshot"
  echo "ok: $dest matches $checksums"
  exit 0
fi

if [[ "$mode" == "fetch" && -d "$dest" ]]; then
  if check_sums; then
    echo "ok: $dest already present and verified"
    exit 0
  fi
  echo "$dest/ exists but does not match $checksums; re-downloading" >&2
fi

command -v curl >/dev/null 2>&1 || die "curl is required"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
zip="$tmp/ml-latest.zip"

echo "downloading $DATA_URL"
if ! curl -fL --retry 3 --retry-delay 2 --progress-bar -o "$zip" "$DATA_URL"; then
  die "download failed from $DATA_URL. The tracked data/raw package is the
authoritative working copy. If you obtain an archive from $UPSTREAM_URL or
another source, verify every file against $checksums and pass it explicitly with:
DATA_URL=file:///abs/path/ml-latest.zip $0"
fi

extract "$zip"
[[ -d "$dest" ]] || die "archive did not contain a top-level $dest/ directory"

check_sums || die "checksum mismatch: $dest does not match the pinned snapshot"
echo "ok: $dest fetched and verified against $checksums"
