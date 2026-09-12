#!/usr/bin/env python3
"""Create reproducible raw-dataset exploration figures and summary metadata."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Callable, Iterable, NoReturn

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as exc:  # pragma: no cover - environment-dependent import guard
    raise SystemExit("explore_dataset.py requires matplotlib and numpy") from exc

try:
    from validate_experiment import ROOT, load_json, validate_experiment
except ImportError as exc:  # pragma: no cover - direct-script import guard
    raise SystemExit("run this script from the repository checkout") from exc

RATING_HEADER = ["userId", "movieId", "rating", "timestamp"]
INTERACTION_HEADER = ["user_id", "item_id"]


def fail(message: str) -> NoReturn:
    raise SystemExit(f"error: {message}")


def repo_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quantile(values: Iterable[int], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def quantile_summary(values: Iterable[int]) -> dict[str, float]:
    materialized = list(values)
    return {
        str(fraction): quantile(materialized, fraction)
        for fraction in (0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1)
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/milestone_1_oracle_comparison.json"),
    )
    parser.add_argument("--manifest", type=Path, default=Path("data/dataset_manifest.json"))
    parser.add_argument(
        "--package-manifest",
        type=Path,
        default=Path("data/chunk_manifest.json"),
    )
    parser.add_argument("--ratings-dir", type=Path, default=Path("data/raw/ratings"))
    parser.add_argument("--movies", type=Path, default=Path("data/raw/movies.csv"))
    parser.add_argument("--cluster-map", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--metadata-root", type=Path, default=Path("results/explorations"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--support-probe-users", type=int)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--random-users", type=int, default=200)
    parser.add_argument("--head-items", type=int, default=200)
    parser.add_argument("--mid-item-rank-start", type=int, default=1001)
    parser.add_argument("--mid-item-rank-end", type=int, default=5000)
    return parser.parse_args()


def rating_chunks(ratings_dir: Path) -> list[Path]:
    chunks = sorted(ratings_dir.glob("ratings-*.csv"))
    if not chunks:
        fail(f"no ratings chunks found under {ratings_dir}")
    return chunks


def iter_ratings(chunks: list[Path]):
    for chunk_index, path in enumerate(chunks):
        try:
            handle = path.open("r", encoding="utf-8", newline="")
        except OSError as exc:
            fail(f"cannot open ratings chunk {path}: {exc}")
        with handle:
            reader = csv.reader(handle)
            if chunk_index == 0:
                try:
                    header = next(reader)
                except StopIteration:
                    fail(f"ratings chunk is empty: {path}")
                if header != RATING_HEADER:
                    fail(f"unexpected ratings header in {path}: {header!r}")
            for line_number, row in enumerate(reader, start=2 if chunk_index == 0 else 1):
                if len(row) != 4:
                    fail(f"invalid ratings row at {path}:{line_number}: {row!r}")
                try:
                    yield int(row[0]), int(row[1]), float(row[2]), int(row[3])
                except (TypeError, ValueError) as exc:
                    fail(f"invalid ratings row at {path}:{line_number}: {exc}")


def read_movies(path: Path) -> set[int]:
    try:
        handle = path.open("r", encoding="utf-8", newline="")
    except OSError as exc:
        fail(f"cannot open movie metadata {path}: {exc}")
    movie_ids: set[int] = set()
    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["movieId", "title", "genres"]:
            fail(f"unexpected movie metadata header: {reader.fieldnames!r}")
        for line_number, row in enumerate(reader, start=2):
            try:
                movie_ids.add(int(row["movieId"]))
            except (TypeError, ValueError) as exc:
                fail(f"invalid movie metadata at {path}:{line_number}: {exc}")
    return movie_ids


def read_cluster_map(path: Path, cluster_count: int) -> dict[int, int]:
    try:
        handle = path.open("r", encoding="utf-8", newline="")
    except OSError as exc:
        fail(f"cannot open cluster map {path}: {exc}")
    assignments: dict[int, int] = {}
    with handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or []) != {"user_id", "cluster_id"}:
            fail("cluster map must have exactly user_id and cluster_id columns")
        for line_number, row in enumerate(reader, start=2):
            try:
                user_id = int(row["user_id"])
                cluster_id = int(row["cluster_id"])
            except (TypeError, ValueError) as exc:
                fail(f"invalid cluster map at {path}:{line_number}: {exc}")
            if cluster_id < 0 or cluster_id >= cluster_count:
                fail(
                    f"cluster {cluster_id} at {path}:{line_number} is outside "
                    f"0 through {cluster_count - 1}"
                )
            previous = assignments.get(user_id)
            if previous is not None and previous != cluster_id:
                fail(f"user {user_id} has conflicting cluster assignments")
            assignments[user_id] = cluster_id
    return assignments


def scan_raw_ratings(chunks: list[Path]) -> dict[str, Any]:
    user_counts: Counter[int] = Counter()
    item_counts: Counter[int] = Counter()
    rating_counts: Counter[float] = Counter()
    user_ids: set[int] = set()
    item_ids: set[int] = set()
    rows = 0
    min_timestamp: int | None = None
    max_timestamp: int | None = None
    previous_user: int | None = None
    for user_id, item_id, rating, timestamp in iter_ratings(chunks):
        if previous_user is not None and user_id < previous_user:
            fail("ratings chunks are not ordered by userId")
        previous_user = user_id
        rows += 1
        user_ids.add(user_id)
        item_ids.add(item_id)
        user_counts[user_id] += 1
        item_counts[item_id] += 1
        rating_counts[rating] += 1
        min_timestamp = timestamp if min_timestamp is None else min(min_timestamp, timestamp)
        max_timestamp = timestamp if max_timestamp is None else max(max_timestamp, timestamp)
    if rows == 0:
        fail("ratings package contains no rows")
    return {
        "rows": rows,
        "user_counts": user_counts,
        "item_counts": item_counts,
        "rating_counts": rating_counts,
        "user_ids": user_ids,
        "item_ids": item_ids,
        "min_timestamp": min_timestamp,
        "max_timestamp": max_timestamp,
    }


def scan_user_histories(
    chunks: list[Path],
    on_row: Callable[[int, int, float, int], None] | None,
    on_user: Callable[[int, list[tuple[int, int, float]]], None],
) -> None:
    current_user: int | None = None
    history: list[tuple[int, int, float]] = []
    for user_id, item_id, rating, timestamp in iter_ratings(chunks):
        if on_row is not None:
            on_row(user_id, item_id, rating, timestamp)
        if current_user is None:
            current_user = user_id
        elif user_id != current_user:
            on_user(current_user, history)
            current_user = user_id
            history = []
        history.append((timestamp, item_id, rating))
    if current_user is not None:
        on_user(current_user, history)


def seeded_tie_key(seed: int, user_id: int, item_id: int) -> bytes:
    value = f"{seed}:{user_id}:{item_id}".encode("utf-8")
    return hashlib.sha256(value).digest()

def split_history(
    user_id: int,
    history: list[tuple[int, int, float]],
    train_fraction: float,
    validation_fraction: float,
    tie_break_seed: int,
) -> tuple[list[tuple[int, int, float]], list[tuple[int, int, float]], list[tuple[int, int, float]]]:
    ordered = sorted(
        history,
        key=lambda row: (row[0], seeded_tie_key(tie_break_seed, user_id, row[1])),
    )
    train_end = math.floor(train_fraction * len(ordered))
    validation_end = math.floor((train_fraction + validation_fraction) * len(ordered))
    return ordered[:train_end], ordered[train_end:validation_end], ordered[validation_end:]


def build_masks(
    chunks: list[Path],
    user_counts: Counter[int],
    item_counts: Counter[int],
    seed: int,
    random_user_count: int,
    head_item_count: int,
    mid_item_rank_start: int,
    mid_item_rank_end: int,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    user_ranked = sorted(user_counts, key=lambda user_id: (-user_counts[user_id], user_id))
    item_ranked = sorted(item_counts, key=lambda item_id: (-item_counts[item_id], item_id))
    if random_user_count > len(user_ranked):
        fail("random user sample is larger than the user population")
    if head_item_count > len(item_ranked):
        fail("head item sample is larger than the item population")
    if not 1 <= mid_item_rank_start <= mid_item_rank_end <= len(item_ranked):
        fail("mid-tail item rank range is outside the popularity-ranked item population")
    rng = random.Random(seed)
    random_users = rng.sample(user_ranked, random_user_count)
    user_rank = {user_id: rank for rank, user_id in enumerate(user_ranked)}
    random_users.sort(key=lambda user_id: user_rank[user_id])
    top_users = user_ranked[:random_user_count]
    head_items = item_ranked[:head_item_count]
    item_rank = {item_id: rank for rank, item_id in enumerate(item_ranked)}
    mid_pool = item_ranked[mid_item_rank_start - 1 : mid_item_rank_end]
    mid_items = rng.sample(mid_pool, head_item_count)
    mid_items.sort(key=lambda item_id: item_rank[item_id])

    panel_specs = {
        "random_head": (random_users, head_items),
        "top_head": (top_users, head_items),
        "random_mid_tail": (random_users, mid_items),
    }
    masks: dict[str, np.ndarray] = {}
    row_maps: dict[str, dict[int, int]] = {}
    col_maps: dict[str, dict[int, int]] = {}
    for name, (users, items) in panel_specs.items():
        masks[name] = np.zeros((len(users), len(items)), dtype=np.uint8)
        row_maps[name] = {user_id: row for row, user_id in enumerate(users)}
        col_maps[name] = {item_id: col for col, item_id in enumerate(items)}

    def fill_mask(user_id: int, item_id: int, rating: float, timestamp: int) -> None:
        del rating, timestamp
        for name in panel_specs:
            row = row_maps[name].get(user_id)
            column = col_maps[name].get(item_id)
            if row is not None and column is not None:
                masks[name][row, column] = 1

    scan_user_histories(chunks, fill_mask, lambda user_id, history: None)
    metadata = {
        name: {
            "user_count": len(users),
            "user_selection": (
                "top_activity_by_raw_rating_count"
                if name == "top_head"
                else "uniform_without_replacement_all_users"
            ),
            "user_axis_order": "descending_raw_rating_count_then_user_id",
            "item_count": len(items),
            "item_selection": (
                "top_items_by_raw_rating_count"
                if name != "random_mid_tail"
                else "uniform_without_replacement_from_item_rank_band"
            ),
            "item_axis_order": "descending_raw_rating_count_then_item_id",
            "item_rank_range_1_indexed": (
                [1, head_item_count]
                if name != "random_mid_tail"
                else [mid_item_rank_start, mid_item_rank_end]
            ),
            "density": float(masks[name].mean()),
        }
        for name, (users, items) in panel_specs.items()
    }
    return masks, metadata


def make_plot_directory(output_dir: Path) -> Path:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    return figures


def render_plots(
    figures: Path,
    raw: dict[str, Any],
    split: dict[str, Any],
    masks: dict[str, np.ndarray],
    mask_metadata: dict[str, dict[str, Any]],
    dimensions: dict[str, int | float],
    pilot_probe: dict[str, Any],
) -> list[str]:
    rows = raw["rows"]
    positive_rows = sum(count for rating, count in raw["rating_counts"].items() if rating >= 4.0)
    user_total_counts = list(raw["user_counts"].values())
    item_count_values = list(raw["item_counts"].values())
    values = sorted(raw["rating_counts"])
    counts = [raw["rating_counts"][value] for value in values]

    figure_paths: list[str] = []
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    colors = ["#4C78A8" if value < 4.0 else "#F58518" for value in values]
    ax.bar(values, counts, width=0.38, color=colors, edgecolor="white", linewidth=0.5)
    ax.axvline(3.75, color="#D62728", linestyle="--", linewidth=1.3, label="positive rule: rating ≥ 4.0")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Rows")
    ax.set_title("Rating values in the raw MovieLens snapshot")
    ax.set_xticks(values)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.legend(frameon=False, loc="upper right")
    ax.text(
        0.02,
        0.95,
        f"{rows:,} rows\npositive rows: {positive_rows:,} ({positive_rows / rows:.1%})",
        transform=ax.transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )
    path = figures / "01_rating_distribution.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    figure_paths.append(display_path(path))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.3))
    user_bins = np.logspace(0, math.log10(max(user_total_counts)), 36)
    item_bins = np.logspace(0, math.log10(max(item_count_values)), 36)
    ax1.hist(user_total_counts, bins=user_bins, color="#4C78A8", edgecolor="white")
    ax2.hist(item_count_values, bins=item_bins, color="#72B7B2", edgecolor="white")
    for axis, title, xlabel in (
        (ax1, "User history length", "ratings per user"),
        (ax2, "Movie popularity", "ratings per movie"),
    ):
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Number of entities")
        axis.set_title(title)
        axis.grid(True, which="both", alpha=0.2)
    ax1.axvline(20, color="#D62728", linestyle="--", linewidth=1, label="20 raw ratings")
    ax1.legend(frameon=False, fontsize=8)
    fig.suptitle("Heavy tails in users and movies", y=1.01, fontsize=12)
    path = figures / "02_user_item_distributions.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    figure_paths.append(display_path(path))

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.9))
    titles = {
        "random_head": "random users; item ranks 1–200",
        "top_head": "top-activity users; item ranks 1–200",
        "random_mid_tail": "same random users; item ranks 1,001–5,000",
    }
    global_density = dimensions["observed_density"]
    for axis, name in zip(axes, ("random_head", "top_head", "random_mid_tail")):
        axis.imshow(masks[name], aspect="auto", interpolation="none", cmap="Greys", vmin=0, vmax=1)
        axis.set_title(titles[name], fontsize=10)
        axis.set_xlabel("items ordered by popularity rank", fontsize=8)
        axis.set_ylabel("users ordered by activity rank", fontsize=8)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.text(
            0.02,
            -0.17,
            f"panel density: {masks[name].mean():.2%}\nglobal observed: {global_density:.3%}",
            transform=axis.transAxes,
            fontsize=8,
            va="top",
        )
    fig.suptitle("Observed-interaction masks; white means no observed rating", y=1.02, fontsize=12)
    path = figures / "03_sparsity_heatmaps.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    figure_paths.append(display_path(path))

    eligible = split["eligible_user_train_positive_counts"]
    eligible_validation = split["eligible_user_validation_positive_counts"]
    eligible_test = split["eligible_user_test_positive_counts"]
    max_x = max(1, int(quantile(eligible, 0.995)))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.5))
    bins = np.arange(0, max_x + 2) - 0.5
    for values_, label, color in (
        (eligible, "train positives", "#4C78A8"),
        (eligible_validation, "validation positives", "#F2CF5B"),
        (eligible_test, "test positives", "#F58518"),
    ):
        ax1.hist(values_, bins=bins, alpha=0.55, label=label, color=color, log=True)
    ax1.set_xlim(-0.5, max_x + 0.5)
    ax1.set_xlabel("Positive interactions per user")
    ax1.set_ylabel("Eligible users (log scale)")
    ax1.set_title("Eligible pilot pool: floor 80/10/10 split")
    ax1.legend(frameon=False, fontsize=8)
    ax1.grid(axis="y", alpha=0.2)
    ax1.text(
        0.03,
        0.95,
        "eligible users: "
        f"{len(eligible):,}\n"
        f"test mean / median: {sum(eligible_test) / len(eligible_test):.1f} / {median(eligible_test):.0f}\n"
        f"test p25–p75: {quantile(eligible_test, 0.25):.0f}–{quantile(eligible_test, 0.75):.0f}\n"
        f"zero test positives: {sum(value == 0 for value in eligible_test):,}",
        transform=ax1.transAxes,
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )
    labels = ["users", "rated\nmovies", "ratings", "positive\nrows"]
    values = [dimensions["user_count"], dimensions["item_count"], rows, positive_rows]
    ax2.bar(labels, values, color=["#4C78A8", "#72B7B2", "#F2CF5B", "#F58518"])
    ax2.set_yscale("log")
    ax2.set_ylabel("Count (log scale)")
    ax2.set_title("Logical data dimensions")
    for index, value in enumerate(values):
        ax2.text(index, value * 1.15, f"{value:,}", ha="center", va="bottom", fontsize=8)
    ax2.text(
        0.02,
        0.04,
        f"dense cells: {dimensions['dense_cells']:,}\n"
        f"observed density: {dimensions['observed_density']:.3%}\n"
        f"positive density: {dimensions['positive_density']:.3%}",
        transform=ax2.transAxes,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )
    fig.suptitle("What the first ranking task will inherit from the data", y=1.01, fontsize=12)
    path = figures / "04_split_and_dimensions.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    figure_paths.append(display_path(path))
    support_values = pilot_probe["item_support_values"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    support_max = max(support_values, default=0)
    bins = np.logspace(0, math.log10(max(2, support_max)), 36)
    ax.hist(support_values, bins=bins, color="#72B7B2", edgecolor="white", linewidth=0, log=True)
    ax.set_xscale("log")
    ax.axvline(
        pilot_probe["support_threshold"],
        color="#D62728",
        linestyle="--",
        linewidth=1.3,
        label=f"support threshold: {pilot_probe['support_threshold']}",
    )
    ax.set_xlabel("Positive training interactions per item")
    ax.set_ylabel("Items (log scale)")
    ax.set_title(f"Global support probe: one {pilot_probe['user_count']:,}-user draw")
    ax.legend(frameon=False, fontsize=8)
    ax.text(
        0.03,
        0.95,
        f"eligible items: {pilot_probe['eligible_item_count']:,}\n"
        f"observed items: {pilot_probe['observed_item_count']:,}",
        transform=ax.transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )
    path = figures / "05_global_support_probe.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    figure_paths.append(display_path(path))


    dashboard_sources = [figures / name for name in (
        "01_rating_distribution.png",
        "02_user_item_distributions.png",
        "03_sparsity_heatmaps.png",
        "04_split_and_dimensions.png",
        "05_global_support_probe.png",
    )]
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    for axis in axes.flat:
        axis.axis("off")
    for axis, source in zip(axes.flat, dashboard_sources):
        axis.imshow(plt.imread(source))
    path = figures / "00_dashboard.png"
    fig.tight_layout(pad=0.5)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    figure_paths.insert(0, display_path(path))
    return figure_paths


def main() -> int:
    args = parse_args()
    config_path = repo_path(args.config)
    manifest_path = repo_path(args.manifest)
    package_manifest_path = repo_path(args.package_manifest)
    ratings_dir = repo_path(args.ratings_dir)
    movies_path = repo_path(args.movies)
    output_dir = repo_path(args.output_root) / args.run_id
    metadata_path = repo_path(args.metadata_root) / f"{args.run_id}.json"
    validate_experiment(config_path, manifest_path)
    config = load_json(config_path)
    objective = load_json(repo_path(Path(config["objective"]["path"])))
    manifest = load_json(manifest_path)
    package_manifest = load_json(package_manifest_path)
    chunks = rating_chunks(ratings_dir)
    movie_ids = read_movies(movies_path)
    cluster_count = config["pilot"]["oracle_clusters"]["count"]
    assignments = None
    cluster_map_path = repo_path(args.cluster_map) if args.cluster_map else None
    if cluster_map_path is not None:
        assignments = read_cluster_map(cluster_map_path, cluster_count)

    raw = scan_raw_ratings(chunks)
    missing_movie_ids = raw["item_ids"] - movie_ids
    if missing_movie_ids:
        fail(f"ratings reference {len(missing_movie_ids)} movie IDs absent from movies.csv")

    train_fraction = config["split"]["train_fraction"]
    validation_fraction = config["split"]["validation_fraction"]
    tie_break_seed = config["split"]["tie_break_seed"]
    positive_threshold = float(objective["positive_rule"].split(">=")[1].strip())
    minimum_train_positive = config["pilot"]["user_selection"]["minimum_train_interactions_per_user"]
    minimum_test_positive = config["split"]["evaluation_eligibility"]["minimum_positive_test_interactions"]
    train_positive_item_counts: Counter[int] = Counter()
    user_train_positive: list[int] = []
    support_threshold = config["pilot"]["support_filter"]["minimum_interactions_per_item_per_cluster"]
    user_validation_positive: list[int] = []
    user_test_positive: list[int] = []
    eligible_user_ids: list[int] = []
    all_user_train_positive: list[int] = []
    all_user_validation_positive: list[int] = []
    all_user_test_positive: list[int] = []
    timestamp_tie_groups = 0
    timestamp_tie_participating_rows = 0
    timestamp_tie_excess_rows = 0
    users_with_timestamp_ties = 0
    cluster_stats: dict[int, dict[str, Any]] = {
        cluster_id: {
            "user_train_positive": [],
            "user_validation_positive": [],
            "user_test_positive": [],
            "item_support": Counter(),
        }
        for cluster_id in range(cluster_count)
    }
    def summarize_user(user_id: int, history: list[tuple[int, int, float]]) -> None:
        nonlocal timestamp_tie_groups
        nonlocal timestamp_tie_participating_rows
        nonlocal timestamp_tie_excess_rows
        nonlocal users_with_timestamp_ties
        timestamp_counts = Counter(row[0] for row in history)
        user_tie_groups = sum(count > 1 for count in timestamp_counts.values())
        timestamp_tie_groups += user_tie_groups
        timestamp_tie_participating_rows += sum(
            count for count in timestamp_counts.values() if count > 1
        )
        timestamp_tie_excess_rows += sum(
            count - 1 for count in timestamp_counts.values() if count > 1
        )
        users_with_timestamp_ties += user_tie_groups > 0
        train, validation, test = split_history(
            user_id,
            history,
            train_fraction,
            validation_fraction,
            tie_break_seed,
        )
        train_positive = [row for row in train if row[2] >= positive_threshold]
        validation_positive = [row for row in validation if row[2] >= positive_threshold]
        test_positive = [row for row in test if row[2] >= positive_threshold]
        train_count = len(train_positive)
        validation_count = len(validation_positive)
        test_count = len(test_positive)
        all_user_train_positive.append(train_count)
        all_user_validation_positive.append(validation_count)
        all_user_test_positive.append(test_count)
        if train_count >= minimum_train_positive:
            eligible_user_ids.append(user_id)
            user_train_positive.append(train_count)
            user_validation_positive.append(validation_count)
            user_test_positive.append(test_count)
        for _, item_id, _ in train_positive:
            train_positive_item_counts[item_id] += 1
        if assignments is not None:
            if user_id not in assignments:
                fail(f"cluster map has no assignment for rating user {user_id}")
            cluster_id = assignments[user_id]
            cluster_stats[cluster_id]["user_train_positive"].append(train_count)
            cluster_stats[cluster_id]["user_validation_positive"].append(validation_count)
            cluster_stats[cluster_id]["user_test_positive"].append(test_count)
            for _, item_id, _ in train_positive:
                cluster_stats[cluster_id]["item_support"][item_id] += 1

    masks, mask_metadata = build_masks(
        chunks,
        raw["user_counts"],
        raw["item_counts"],
        args.seed,
        args.random_users,
        args.head_items,
        args.mid_item_rank_start,
        args.mid_item_rank_end,
    )
    scan_user_histories(chunks, None, summarize_user)
    if len(user_train_positive) == 0:
        fail("no users satisfy the minimum positive training-interaction threshold")
    probe_user_count = (
        args.support_probe_users
        if args.support_probe_users is not None
        else config["pilot"]["user_selection"]["target_count"] // cluster_count
    )
    if probe_user_count <= 0:
        fail("support probe user count must be positive")
    if len(eligible_user_ids) < probe_user_count:
        fail(
            f"only {len(eligible_user_ids)} users satisfy the positive training threshold; "
            f"need {probe_user_count} for the global support probe"
        )
    probe_users = sorted(
        random.Random(args.seed).sample(sorted(eligible_user_ids), probe_user_count)
    )
    probe_user_set = set(probe_users)
    probe_item_support: Counter[int] = Counter()

    def collect_probe(user_id: int, history: list[tuple[int, int, float]]) -> None:
        if user_id not in probe_user_set:
            return
        train, _, _ = split_history(
            user_id,
            history,
            train_fraction,
            validation_fraction,
            tie_break_seed,
        )
        for _, item_id, rating in train:
            if rating >= positive_threshold:
                probe_item_support[item_id] += 1

    scan_user_histories(chunks, None, collect_probe)
    pilot_support_probe = {
        "user_count": probe_user_count,
        "seed": args.seed,
        "population": "uniform_without_replacement_users_with_minimum_positive_train_interactions",
        "minimum_train_positive_interactions": minimum_train_positive,
        "support_threshold": support_threshold,
        "positive_train_interaction_count": sum(probe_item_support.values()),
        "observed_item_count": len(probe_item_support),
        "eligible_item_count": sum(
            count >= support_threshold for count in probe_item_support.values()
        ),
        "item_support_quantiles": quantile_summary(probe_item_support.values()),
        "item_support_values": list(probe_item_support.values()),
    }

    dense_cells = len(raw["user_ids"]) * len(raw["item_ids"])
    positive_rows = sum(count for rating, count in raw["rating_counts"].items() if rating >= positive_threshold)
    dimensions = {
        "user_count": len(raw["user_ids"]),
        "item_count": len(raw["item_ids"]),
        "movie_metadata_count": len(movie_ids),
        "rating_count": raw["rows"],
        "positive_rating_count": positive_rows,
        "dense_cells": dense_cells,
        "observed_density": raw["rows"] / dense_cells,
        "positive_density": positive_rows / dense_cells,
    }
    evaluation_eligible_test = [
        value for value in user_test_positive if value >= minimum_test_positive
    ]
    split_summary: dict[str, Any] = {
        "partition_object": config["split"]["partition_object"],
        "ordering": config["split"]["ordering"],
        "tie_break": config["split"]["tie_break"],
        "tie_break_seed": tie_break_seed,
        "boundary_rounding": config["split"]["boundary_rounding"],
        "positive_filter_timing": config["split"]["positive_filter_timing"],
        "train_fraction": train_fraction,
        "validation_fraction": validation_fraction,
        "test_fraction": config["split"]["test_fraction"],
        "positive_threshold": positive_threshold,
        "minimum_train_positive_interactions": minimum_train_positive,
        "minimum_test_positive_interactions": minimum_test_positive,
        "timestamp_tie_groups": timestamp_tie_groups,
        "timestamp_tie_participating_rows": timestamp_tie_participating_rows,
        "timestamp_tie_excess_rows": timestamp_tie_excess_rows,
        "users_with_timestamp_ties": users_with_timestamp_ties,
        "timestamp_tie_excess_fraction_of_rows": timestamp_tie_excess_rows / raw["rows"],
        "all_user_test_positive_quantiles": quantile_summary(all_user_test_positive),
        "eligible_user_count": len(user_train_positive),
        "eligible_user_train_positive_quantiles": quantile_summary(user_train_positive),
        "eligible_user_validation_positive_quantiles": quantile_summary(user_validation_positive),
        "eligible_user_test_positive_quantiles": quantile_summary(user_test_positive),
        "eligible_user_zero_test_positive_count": sum(value == 0 for value in user_test_positive),
        "evaluation_eligible_user_count": len(evaluation_eligible_test),
        "evaluation_excluded_zero_test_positive_count": len(user_test_positive) - len(evaluation_eligible_test),
        "all_user_train_positive_counts": all_user_train_positive,
        "all_user_validation_positive_counts": all_user_validation_positive,
        "all_user_test_positive_counts": all_user_test_positive,
        "eligible_user_train_positive_counts": user_train_positive,
        "eligible_user_validation_positive_counts": user_validation_positive,
        "eligible_user_test_positive_counts": user_test_positive,
    }
    figures_dir = make_plot_directory(output_dir)
    figure_paths = render_plots(
        figures_dir,
        raw,
        split_summary,
        masks,
        mask_metadata,
        dimensions,
        pilot_support_probe,
    )
    split_output = {
        key: value for key, value in split_summary.items() if not key.endswith("_counts")
    }
    pilot_support_probe_output = {
        key: value
        for key, value in pilot_support_probe.items()
        if key != "item_support_values"
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    cluster_summary: dict[str, Any] | None = None
    if assignments is not None:
        cluster_summary = {}
        eligible_items_by_cluster: dict[int, set[int]] = {}
        for cluster_id, values in cluster_stats.items():
            support: Counter[int] = values["item_support"]
            eligible_items = {item_id for item_id, count in support.items() if count >= support_threshold}
            eligible_items_by_cluster[cluster_id] = eligible_items
            cluster_summary[str(cluster_id)] = {
                "mapped_user_count": len(values["user_train_positive"]),
                "eligible_train_user_count": sum(
                    count >= minimum_train_positive for count in values["user_train_positive"]
                ),
                "evaluation_eligible_user_count": sum(
                    train_count >= minimum_train_positive and test_count >= minimum_test_positive
                    for train_count, test_count in zip(
                        values["user_train_positive"], values["user_test_positive"]
                    )
                ),
                "positive_train_interaction_count": sum(support.values()),
                "eligible_item_count_at_support_threshold": len(eligible_items),
                "support_threshold": support_threshold,
                "item_support_quantiles": quantile_summary(support.values()),
            }
        cluster_summary["all_cluster_intersection_eligible_item_count"] = len(
            set.intersection(*eligible_items_by_cluster.values())
            if eligible_items_by_cluster
            else set()
        )

    figure_artifacts = [
        {"path": figure, "sha256": sha256_file(repo_path(Path(figure)))}
        for figure in figure_paths
    ]

    summary = {
        "run_id": args.run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "script": {
            "path": display_path(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "pilot_support_probe": pilot_support_probe_output,
        "inputs": {
            "config_path": display_path(config_path),
            "config_sha256": sha256_file(config_path),
            "manifest_path": display_path(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "package_manifest_path": display_path(package_manifest_path),
            "package_manifest_sha256": sha256_file(package_manifest_path),
            "dataset_id": manifest["dataset_id"],
            "dataset_version": manifest["version"],
            "ratings_chunks": [display_path(path) for path in chunks],
            "movies_path": display_path(movies_path),
            "cluster_map_path": display_path(cluster_map_path) if cluster_map_path else None,
            "cluster_map_sha256": sha256_file(cluster_map_path) if cluster_map_path else None,
        },
        "chunk_headers": {
            display_path(path): path.open("r", encoding="utf-8", newline="").readline().rstrip("\r\n")
            for path in chunks
        },
        "dimensions": dimensions,
        "ratings": {
            "count_by_value": {str(rating): count for rating, count in sorted(raw["rating_counts"].items())},
            "timestamp_range": [raw["min_timestamp"], raw["max_timestamp"]],
            "user_count_quantiles": quantile_summary(raw["user_counts"].values()),
            "item_count_quantiles": quantile_summary(raw["item_counts"].values()),
            "positive_train_item_support_quantiles": quantile_summary(train_positive_item_counts.values()),
            "positive_train_item_support_threshold": support_threshold,
            "positive_train_items_at_support_threshold": sum(
                count >= support_threshold for count in train_positive_item_counts.values()
            ),
        },
        "split": split_output,
        "sparsity_sampling": {
            "seed": args.seed,
            "heatmap_shape": [args.random_users, args.head_items],
            "user_axis_order": "descending_raw_rating_count_then_user_id",
            "item_axis_order": "descending_raw_rating_count_then_item_id",
            "views": mask_metadata,
        },
        "figures": figure_artifacts,
        "cluster_summary": cluster_summary,
    }
    summary_path = metadata_path
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"created {display_path(summary_path)}")
    for figure in figure_paths:
        print(f"created {figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
