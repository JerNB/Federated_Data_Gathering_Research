#!/usr/bin/env python3
"""Build a queryable SQLite package from the pinned MovieLens snapshot.

The raw CSV files remain the source of truth. This command creates a derived
package for local modeling or upload to a public dataset registry. The output
is reproducible from the input directory, the manifest, and the command line.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, NoReturn, Sequence

BATCH_SIZE = 10_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("ml-latest"),
        help="directory containing the raw MovieLens CSV files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data_working/movielens.sqlite"),
        help="derived SQLite path",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/dataset_manifest.json"),
        help="source dataset manifest",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="limit rows imported from each source table for a smoke build",
    )
    parser.add_argument(
        "--include-genome",
        action="store_true",
        help="include genome tags and genome scores in the derived package",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing output file",
    )
    return parser.parse_args()


def fail(message: str) -> NoReturn:
    raise SystemExit(f"error: {message}")


def read_manifest(path: Path) -> tuple[dict, str]:
    try:
        raw = path.read_bytes()
        manifest = json.loads(raw)
    except OSError as exc:
        fail(f"cannot read manifest {path}: {exc}")
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON in manifest {path}: {exc}")
    if manifest.get("dataset_id") != "movielens_ml_latest":
        fail("manifest dataset_id must be movielens_ml_latest")
    return manifest, hashlib.sha256(raw).hexdigest()


def source_rows(
    path: Path,
    expected_header: Sequence[str],
    limit: int | None,
) -> Iterable[list[str]]:
    try:
        handle = path.open("r", encoding="utf-8", newline="")
    except OSError as exc:
        fail(f"cannot open {path}: {exc}")
    with handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header != list(expected_header):
            fail(f"unexpected header in {path}: {header!r}")
        for row_number, row in enumerate(reader, start=1):
            if len(row) != len(expected_header):
                fail(f"wrong column count in {path} row {row_number}")
            yield row
            if limit is not None and row_number >= limit:
                return


def insert_rows(
    connection: sqlite3.Connection,
    table: str,
    columns: Sequence[str],
    rows: Iterable[Sequence[object]],
) -> int:
    placeholders = ",".join("?" for _ in columns)
    statement = f"INSERT INTO {table} ({','.join(columns)}) VALUES ({placeholders})"
    batch: list[Sequence[object]] = []
    count = 0
    for row in rows:
        batch.append(row)
        if len(batch) == BATCH_SIZE:
            connection.executemany(statement, batch)
            count += len(batch)
            batch.clear()
    if batch:
        connection.executemany(statement, batch)
        count += len(batch)
    return count


def create_schema(connection: sqlite3.Connection, include_genome: bool) -> None:
    connection.executescript(
        """
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE movies (
            movie_id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            genres TEXT NOT NULL
        );
        CREATE TABLE links (
            movie_id INTEGER PRIMARY KEY,
            imdb_id TEXT,
            tmdb_id TEXT
        );
        CREATE TABLE ratings (
            user_id INTEGER NOT NULL,
            movie_id INTEGER NOT NULL,
            rating REAL NOT NULL,
            timestamp INTEGER NOT NULL
        );
        CREATE TABLE tags (
            user_id INTEGER NOT NULL,
            movie_id INTEGER NOT NULL,
            tag TEXT NOT NULL,
            timestamp INTEGER NOT NULL
        );
        """
    )
    if include_genome:
        connection.executescript(
            """
            CREATE TABLE genome_tags (
                tag_id INTEGER PRIMARY KEY,
                tag TEXT NOT NULL
            );
            CREATE TABLE genome_scores (
                movie_id INTEGER NOT NULL,
                tag_id INTEGER NOT NULL,
                relevance REAL NOT NULL
            );
            """
        )


def add_indexes(connection: sqlite3.Connection, include_genome: bool) -> None:
    connection.executescript(
        """
        CREATE INDEX ratings_user_id ON ratings (user_id);
        CREATE INDEX ratings_movie_id ON ratings (movie_id);
        CREATE INDEX ratings_timestamp ON ratings (timestamp);
        CREATE INDEX links_imdb_id ON links (imdb_id);
        CREATE INDEX tags_user_id ON tags (user_id);
        CREATE INDEX tags_movie_id ON tags (movie_id);
        """
    )
    if include_genome:
        connection.executescript(
            """
            CREATE INDEX genome_scores_movie_id ON genome_scores (movie_id);
            CREATE INDEX genome_scores_tag_id ON genome_scores (tag_id);
            """
        )


def populate(
    connection: sqlite3.Connection,
    input_dir: Path,
    limit: int | None,
    include_genome: bool,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    table_specs = [
        (
            "movies.csv",
            "movies",
            ("movieId", "title", "genres"),
            ("movie_id", "title", "genres"),
            lambda row: (int(row[0]), row[1], row[2]),
        ),
        (
            "links.csv",
            "links",
            ("movieId", "imdbId", "tmdbId"),
            ("movie_id", "imdb_id", "tmdb_id"),
            lambda row: (int(row[0]), row[1] or None, row[2] or None),
        ),
        (
            "ratings.csv",
            "ratings",
            ("userId", "movieId", "rating", "timestamp"),
            ("user_id", "movie_id", "rating", "timestamp"),
            lambda row: (int(row[0]), int(row[1]), float(row[2]), int(row[3])),
        ),
        (
            "tags.csv",
            "tags",
            ("userId", "movieId", "tag", "timestamp"),
            ("user_id", "movie_id", "tag", "timestamp"),
            lambda row: (int(row[0]), int(row[1]), row[2], int(row[3])),
        ),
    ]
    if include_genome:
        table_specs.extend(
            [
                (
                    "genome-tags.csv",
                    "genome_tags",
                    ("tagId", "tag"),
                    ("tag_id", "tag"),
                    lambda row: (int(row[0]), row[1]),
                ),
                (
                    "genome-scores.csv",
                    "genome_scores",
                    ("movieId", "tagId", "relevance"),
                    ("movie_id", "tag_id", "relevance"),
                    lambda row: (int(row[0]), int(row[1]), float(row[2])),
                ),
            ]
        )

    for filename, table, header, columns, transform in table_specs:
        print(f"loading {filename}", file=sys.stderr)
        rows = (transform(row) for row in source_rows(input_dir / filename, header, limit))
        counts[table] = insert_rows(connection, table, columns, rows)
        connection.commit()
        print(f"loaded {counts[table]:,} rows into {table}", file=sys.stderr)
    return counts


def write_metadata(
    connection: sqlite3.Connection,
    manifest: dict,
    manifest_sha256: str,
    counts: dict[str, int],
    include_genome: bool,
    limit: int | None,
) -> None:
    values = {
        "dataset_id": manifest["dataset_id"],
        "dataset_version": manifest["version"],
        "source_snapshot_date": manifest["source"]["snapshot_date"],
        "source_archive_sha256": manifest["archive"]["sha256"],
        "source_manifest_sha256": manifest_sha256,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "include_genome": json.dumps(include_genome),
        "row_limit_per_table": "all" if limit is None else str(limit),
        "row_counts": json.dumps(counts, sort_keys=True),
    }
    connection.executemany(
        "INSERT INTO metadata (key, value) VALUES (?, ?)", values.items()
    )
    connection.commit()


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        fail("--limit must be positive")
    if not args.input.is_dir():
        fail(f"input directory not found: {args.input}")
    if args.output.exists() and not args.force:
        fail(f"output exists: {args.output}; use --force to replace it")

    manifest, manifest_sha256 = read_manifest(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(
        prefix=f".{args.output.name}.",
        suffix=".tmp",
        dir=args.output.parent,
        delete=False,
    )
    temporary_path = Path(temporary.name)
    temporary.close()

    try:
        connection = sqlite3.connect(temporary_path)
        try:
            connection.execute("PRAGMA journal_mode = OFF")
            connection.execute("PRAGMA synchronous = OFF")
            connection.execute("PRAGMA temp_store = MEMORY")
            create_schema(connection, args.include_genome)
            counts = populate(connection, args.input, args.limit, args.include_genome)
            add_indexes(connection, args.include_genome)
            write_metadata(
                connection,
                manifest,
                manifest_sha256,
                counts,
                args.include_genome,
                args.limit,
            )
            connection.execute("PRAGMA optimize")
        finally:
            connection.close()
        os.replace(temporary_path, args.output)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    print(f"created {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
