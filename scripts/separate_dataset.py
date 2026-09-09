#!/usr/bin/env python3
"""Create and verify a Git-friendly, byte-preserving CSV package."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import NoReturn

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_MANIFEST = Path("data/dataset_manifest.json")
DEFAULT_CHUNK_MANIFEST = Path("data/chunk_manifest.json")
DEFAULT_OUTPUT_ROOT = Path("data/raw")
DEFAULT_CHUNK_SIZE = 80 * 1024 * 1024


def fail(message: str) -> NoReturn:
    raise SystemExit(f"error: {message}")


def sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(block)
            digest.update(block)
    return size, digest.hexdigest()


def load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        fail(f"cannot read {path}: {exc}")
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON in {path}: {exc}")
    if not isinstance(value, dict):
        fail(f"top level of {path} must be an object")
    return value


def relative_source_path(manifest_path: str) -> Path:
    parts = Path(manifest_path).parts
    if len(parts) < 2 or parts[0] != "ml-latest":
        fail(f"source manifest path must begin with ml-latest/: {manifest_path}")
    return Path(*parts[1:])


def relative_repo_path(path: Path) -> str:
    return str(path.relative_to(ROOT))


def split_large_file(
    source_path: Path,
    output_directory: Path,
    stem: str,
    suffix: str,
    chunk_size: int,
    force: bool,
) -> tuple[list[dict], str, int]:
    output_directory.mkdir(parents=True, exist_ok=True)
    stale = sorted(output_directory.glob(f"{stem}-*{suffix}"))
    if stale and not force:
        fail(f"chunk files already exist in {output_directory}; use --force to replace them")
    if force:
        for path in stale:
            path.unlink()
    source_digest = hashlib.sha256()
    reassembled_digest = hashlib.sha256()
    chunks: list[dict] = []
    total_bytes = 0

    with source_path.open("rb") as source:
        header = source.readline()
        if not header:
            fail(f"cannot split empty file: {source_path}")
        if len(header) > chunk_size:
            fail(f"header exceeds chunk size in {source_path}")
        source_digest.update(header)
        reassembled_digest.update(header)

        part_index = 0
        part_path = output_directory / f"{stem}-{part_index:03d}{suffix}"
        part_handle = part_path.open("wb")
        part_handle.write(header)
        part_bytes = len(header)

        for line in source:
            if len(line) > chunk_size:
                part_handle.close()
                fail(f"row exceeds chunk size in {source_path}")
            if part_bytes + len(line) > chunk_size:
                if part_index == 0 and part_bytes == len(header):
                    part_handle.close()
                    fail(f"header plus first row exceeds chunk size in {source_path}")
                part_handle.close()
                size, digest = sha256_file(part_path)
                chunks.append({"path": relative_repo_path(part_path), "bytes": size, "sha256": digest})
                part_index += 1
                part_path = output_directory / f"{stem}-{part_index:03d}{suffix}"
                part_handle = part_path.open("wb")
                part_bytes = 0
            source_digest.update(line)
            reassembled_digest.update(line)
            part_handle.write(line)
            part_bytes += len(line)
            total_bytes += len(line)
        part_handle.close()
        size, digest = sha256_file(part_path)
        chunks.append({"path": relative_repo_path(part_path), "bytes": size, "sha256": digest})

    source_hex = source_digest.hexdigest()
    if reassembled_digest.hexdigest() != source_hex:
        fail(f"internal reassembly mismatch for {source_path}")
    return chunks, source_hex, len(header) + total_bytes


def copy_small_file(
    source_path: Path,
    output_path: Path,
    force: bool,
) -> tuple[list[dict], str, int]:
    if output_path.exists() and not force:
        fail(f"package file exists: {output_path}; use --force to replace it")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, output_path)
    size, digest = sha256_file(output_path)
    return [{"path": relative_repo_path(output_path), "bytes": size, "sha256": digest}], digest, size


def reassembled_digest(paths: list[Path]) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    for path in paths:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                size += len(block)
                digest.update(block)
    return size, digest.hexdigest()


def build_package(args: argparse.Namespace) -> None:
    source_manifest_path = args.source_manifest
    source_manifest = load_json(source_manifest_path)
    source_manifest_digest = sha256_file(source_manifest_path)[1]
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    files = []

    for source_entry in source_manifest["files"]:
        source_relative = relative_source_path(source_entry["path"])
        source_path = args.input / source_relative
        if not source_path.is_file():
            fail(f"source file not found: {source_path}")
        source_size = source_path.stat().st_size
        if source_size > args.chunk_size:
            output_directory = output_root / source_relative.with_suffix("")
            chunks, source_digest, reassembled_bytes = split_large_file(
                source_path,
                output_directory,
                source_relative.stem,
                source_relative.suffix,
                args.chunk_size,
                args.force,
            )
            chunked = True
        else:
            output_path = output_root / source_relative
            chunks, source_digest, reassembled_bytes = copy_small_file(
                source_path,
                output_path,
                args.force,
            )
            chunked = False
        if source_digest != source_entry["sha256"]:
            fail(
                f"source checksum mismatch for {source_path}: "
                f"{source_digest} versus {source_entry['sha256']}"
            )
        files.append(
            {
                "source_path": source_entry["path"],
                "source_bytes": source_size,
                "source_sha256": source_digest,
                "editable": False,
                "chunked": chunked,
                "header_in_first_chunk_only": chunked,
                "reassembled_bytes": reassembled_bytes,
                "reassembled_sha256": source_digest,
                "chunks": chunks,
            }
        )
        print(f"separated {source_entry['path']}")

    package_manifest = {
        "format": "git_csv_package",
        "version": "1",
        "source_dataset_id": source_manifest["dataset_id"],
        "source_version": source_manifest["version"],
        "source_manifest_sha256": source_manifest_digest,
        "output_root": relative_repo_path(output_root),
        "chunk_size_bytes": args.chunk_size,
        "line_endings": "preserved",
        "files": files,
    }
    args.chunk_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.chunk_manifest.write_text(json.dumps(package_manifest, indent=2) + "\n", encoding="utf-8")
    print(f"created {args.chunk_manifest}")


def verify_package(args: argparse.Namespace) -> None:
    package_manifest = load_json(args.chunk_manifest)
    expected_source_manifest = package_manifest["source_manifest_sha256"]
    actual_source_manifest = sha256_file(args.source_manifest)[1]
    if actual_source_manifest != expected_source_manifest:
        fail(f"source manifest checksum mismatch: {args.source_manifest}")
    source_manifest = load_json(args.source_manifest) if args.input.is_dir() else None
    output_root = ROOT / package_manifest["output_root"]
    if not output_root.is_dir():
        fail(f"package output root not found: {output_root}")
    known_paths: set[str] = set()
    for file_entry in package_manifest["files"]:
        chunk_paths = [ROOT / chunk["path"] for chunk in file_entry["chunks"]]
        known_paths.update(str(path.relative_to(ROOT)) for path in chunk_paths)
        for chunk, path in zip(file_entry["chunks"], chunk_paths):
            if not path.is_file():
                fail(f"package file not found: {path}")
            size, digest = sha256_file(path)
            if size != chunk["bytes"] or digest != chunk["sha256"]:
                fail(f"package checksum mismatch for {path}")
        size, digest = reassembled_digest(chunk_paths)
        if size != file_entry["reassembled_bytes"] or digest != file_entry["reassembled_sha256"]:
            fail(f"reassembled checksum mismatch for {file_entry['source_path']}")
        if source_manifest is not None:
            source_path = args.input / relative_source_path(file_entry["source_path"])
            source_size, source_digest = sha256_file(source_path)
            if source_size != file_entry["source_bytes"] or source_digest != file_entry["source_sha256"]:
                fail(f"source checksum mismatch for {source_path}")
    for path in output_root.rglob("*"):
        if not path.is_file() or path.name.startswith("."):
            continue
        if str(path.relative_to(ROOT)) not in known_paths:
            fail(f"unlisted package file: {path}")
    print(f"verified {args.chunk_manifest}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "ml-latest")
    parser.add_argument("--output-root", type=Path, default=ROOT / DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--source-manifest", type=Path, default=ROOT / DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--chunk-manifest", type=Path, default=ROOT / DEFAULT_CHUNK_MANIFEST)
    parser.add_argument("--chunk-size-mib", type=int, default=80)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.input = args.input.resolve()
    args.output_root = args.output_root.resolve()
    args.source_manifest = args.source_manifest.resolve()
    args.chunk_manifest = args.chunk_manifest.resolve()
    args.chunk_size = args.chunk_size_mib * 1024 * 1024
    if args.chunk_size <= 0:
        fail("--chunk-size-mib must be positive")
    if args.verify:
        verify_package(args)
        return 0
    if not args.input.is_dir():
        fail(f"input directory not found: {args.input}")
    if args.chunk_manifest.exists() and not args.force:
        fail(f"manifest exists: {args.chunk_manifest}; use --force to replace it")
    build_package(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
