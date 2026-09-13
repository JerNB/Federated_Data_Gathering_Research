#!/usr/bin/env python3
"""Run repeated full-snapshot sampling experiments with fixed recommenders.

The runner builds one frozen full-data reference, caches the parsed positive
training pool, and reuses that reference for every sampling candidate. The
headline frame is fixed across cells; sample-native exclusion is retained as a
separate sensitivity result. Models are popularity, count-weighted popularity,
and deterministic item-item cosine.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import pickle
import random
import resource
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
MASK64 = (1 << 64) - 1
RATING_THRESHOLD = 4.0
TRAIN_FRACTION = 0.8
VALIDATION_FRACTION = 0.1
TEST_FRACTION = 0.1
TIE_BREAK_SEED = 20260910
BASE_SEED = 20260909
DEFAULT_FRACTIONS = (0.01, 0.025, 0.05, 0.10, 0.25, 0.50, 1.0)
PRIMARY_SCHEMES = (
    "uniform_user",
    "activity_stratified_user",
    "uniform_interaction",
    "within_user_history",
)
MODELS = ("popularity", "rating_weighted_popularity", "item_item_cosine")
REFERENCE_CACHE_VERSION = "sample_generalization_reference_v2"
TRAINING_CACHE_VERSION = "sample_generalization_training_v2"
TRAINING_DTYPE = np.dtype([
    ("user_id", "<i4"),
    ("item_index", "<i4"),
    ("rating", "<f4"),
    ("timestamp", "<i8"),
])


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings-dir", type=Path, default=Path("data/raw/ratings"))
    parser.add_argument("--manifest", type=Path, default=Path("data/dataset_manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("results/sample_generalization"))
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/sample_generalization_cache"))
    parser.add_argument("--fractions", default=",".join(map(str, DEFAULT_FRACTIONS)))
    parser.add_argument("--schemes", default=",".join(PRIMARY_SCHEMES))
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--panel-size", type=int, default=2000)
    parser.add_argument("--cutoff", type=int, default=10)
    parser.add_argument("--smoothing-count", type=float, default=20.0)
    parser.add_argument("--support-threshold", type=int, default=20)
    parser.add_argument("--item-item-top-k", "--knn-neighbors", dest="item_item_top_k", type=int, default=100)
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    return parser.parse_args()


def repo_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def keyed_u64(seed: int, *values: int) -> int:
    value = seed & MASK64
    for item in values:
        value = mix64(value ^ (int(item) & MASK64))
    return value


def tie_key(user_id: int, item_id: int) -> int:
    return keyed_u64(TIE_BREAK_SEED, user_id, item_id)


def stable_seed(experiment: str, scheme: str, replicate: int) -> int:
    payload = f"{experiment}:{scheme}:{replicate}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def rating_chunks(ratings_dir: Path) -> list[Path]:
    chunks = sorted(ratings_dir.glob("ratings-*.csv"))
    if not chunks:
        fail(f"no ratings chunks found under {ratings_dir}")
    return chunks


def iter_rows(chunks: list[Path]) -> Iterator[tuple[int, int, float, int]]:
    previous_user: int | None = None
    for chunk_index, path in enumerate(chunks):
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            if chunk_index == 0:
                header = next(reader, None)
                if header != ["userId", "movieId", "rating", "timestamp"]:
                    fail(f"unexpected ratings header in {path}: {header}")
            for line_number, row in enumerate(reader, 2 if chunk_index == 0 else 1):
                if len(row) != 4:
                    fail(f"invalid ratings row at {path}:{line_number}")
                try:
                    user_id, item_id = int(row[0]), int(row[1])
                    rating, timestamp = float(row[2]), int(row[3])
                except ValueError as exc:
                    fail(f"invalid ratings row at {path}:{line_number}: {exc}")
                if previous_user is not None and user_id < previous_user:
                    fail("ratings chunks are not ordered by userId")
                previous_user = user_id
                yield user_id, item_id, rating, timestamp


def iter_user_histories(chunks: list[Path]) -> Iterator[tuple[int, list[tuple[int, int, float, int]]]]:
    current_user: int | None = None
    history: list[tuple[int, int, float, int]] = []
    for row in iter_rows(chunks):
        user_id = row[0]
        if current_user is None:
            current_user = user_id
        elif user_id != current_user:
            yield current_user, history
            current_user, history = user_id, []
        history.append(row)
    if current_user is not None:
        yield current_user, history


def split_bounds(history: list[tuple[int, int, float, int]]) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    ordered = sorted(history, key=lambda row: (row[3], tie_key(row[0], row[1])))
    train_end = math.floor(TRAIN_FRACTION * len(ordered))
    validation_end = math.floor((TRAIN_FRACTION + VALIDATION_FRACTION) * len(ordered))
    train_last = None if train_end == 0 else (ordered[train_end - 1][3], tie_key(ordered[train_end - 1][0], ordered[train_end - 1][1]))
    validation_last = None if validation_end == 0 else (ordered[validation_end - 1][3], tie_key(ordered[validation_end - 1][0], ordered[validation_end - 1][1]))
    return train_last, validation_last


def classify_row(user_id: int, item_id: int, timestamp: int, bounds: tuple[tuple[int, int] | None, tuple[int, int] | None]) -> str:
    key = (timestamp, tie_key(user_id, item_id))
    train_last, validation_last = bounds
    if train_last is not None and key <= train_last:
        return "train"
    if validation_last is not None and key <= validation_last:
        return "validation"
    return "test"


def reference_cache_key(manifest_sha: str, panel_size: int, seed: int) -> str:
    payload = json.dumps({
        "version": REFERENCE_CACHE_VERSION,
        "manifest_sha256": manifest_sha,
        "panel_size": panel_size,
        "seed": seed,
        "rating_threshold": RATING_THRESHOLD,
        "split": [TRAIN_FRACTION, VALIDATION_FRACTION, TEST_FRACTION],
        "tie_break_seed": TIE_BREAK_SEED,
    }, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:24]


def build_reference(chunks: list[Path], panel_size: int, seed: int) -> dict[str, object]:
    started = time.perf_counter()
    bounds: dict[int, tuple[tuple[int, int] | None, tuple[int, int] | None]] = {}
    train_counts: Counter[int] = Counter()
    test_counts: Counter[int] = Counter()
    item_counts: Counter[int] = Counter()
    item_sums: Counter[int] = Counter()
    user_ids: list[int] = []
    item_ids: set[int] = set()
    rating_counts: Counter[float] = Counter()
    raw_rows = positive_rows = 0
    for user_id, history in iter_user_histories(chunks):
        user_ids.append(user_id)
        bounds[user_id] = split_bounds(history)
        for row_user, item_id, rating, timestamp in history:
            item_ids.add(item_id)
            rating_counts[rating] += 1
            part = classify_row(row_user, item_id, timestamp, bounds[user_id])
            if part == "train" and rating >= RATING_THRESHOLD:
                positive_rows += 1
                train_counts[user_id] += 1
                item_counts[item_id] += 1
                item_sums[item_id] += rating
            elif part == "test" and rating >= RATING_THRESHOLD:
                test_counts[user_id] += 1
            raw_rows += 1
    eligible = [u for u in user_ids if train_counts[u] >= 20 and test_counts[u] >= 1]
    if not eligible:
        fail("no evaluation-eligible users found")
    panel = sorted(random.Random(seed).sample(eligible, min(panel_size, len(eligible))))
    panel_set = set(panel)
    panel_records = {u: {"train": set(), "test": set()} for u in panel}
    for user_id, item_id, rating, timestamp in iter_rows(chunks):
        if user_id not in panel_set or rating < RATING_THRESHOLD:
            continue
        part = classify_row(user_id, item_id, timestamp, bounds[user_id])
        if part in ("train", "test"):
            panel_records[user_id][part].add(item_id)
    ordered_users = sorted(train_counts, key=lambda u: (train_counts[u], u))
    strata = {u: min(9, (rank * 10) // max(1, len(ordered_users))) for rank, u in enumerate(ordered_users)}
    positive_mean = sum(item_sums.values()) / max(1, sum(item_counts.values()))
    return {
        "bounds": bounds,
        "train_positive_counts": train_counts,
        "test_positive_counts": test_counts,
        "full_item_counts": item_counts,
        "full_item_rating_sums": item_sums,
        "user_ids": user_ids,
        "item_ids": sorted(item_ids),
        "rating_counts": rating_counts,
        "raw_rows": raw_rows,
        "positive_rows": positive_rows,
        "eligible_users": eligible,
        "panel": panel,
        "panel_records": panel_records,
        "activity_strata": strata,
        "positive_mean": positive_mean,
        "reference_seconds": time.perf_counter() - started,
        "reference_cache_hit": False,
    }


def load_or_build_reference(chunks: list[Path], manifest_sha: str, panel_size: int, seed: int, cache_dir: Path) -> tuple[dict[str, object], str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = reference_cache_key(manifest_sha, panel_size, seed)
    state_path = cache_dir / f"reference_{key}.pkl"
    metadata_path = cache_dir / f"reference_{key}.json"
    if state_path.is_file() and metadata_path.is_file():
        try:
            reference = pickle.loads(state_path.read_bytes())
            if reference.get("positive_rows") and reference.get("item_ids"):
                reference["reference_cache_hit"] = True
                reference["reference_seconds"] = 0.0
                return reference, key
        except (OSError, EOFError, pickle.PickleError, ValueError):
            pass
    reference = build_reference(chunks, panel_size, seed)
    state_path.write_bytes(pickle.dumps(reference, protocol=pickle.HIGHEST_PROTOCOL))
    metadata_path.write_text(json.dumps({
        "cache_version": REFERENCE_CACHE_VERSION,
        "cache_key": key,
        "manifest_sha256": manifest_sha,
        "panel_size": panel_size,
        "seed": seed,
        "raw_rows": reference["raw_rows"],
        "positive_rows": reference["positive_rows"],
    }, indent=2) + "\n", encoding="utf-8")
    return reference, key


def load_training_cache(chunks: list[Path], reference: dict[str, object], cache_dir: Path, key: str) -> tuple[np.memmap, bool]:
    data_path = cache_dir / f"training_positive_{key}.dat"
    metadata_path = cache_dir / f"training_positive_{key}.json"
    count = int(reference["positive_rows"])
    cache_hit = False
    if data_path.is_file() and metadata_path.is_file() and data_path.stat().st_size == count * TRAINING_DTYPE.itemsize:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            cache_hit = metadata.get("cache_version") == TRAINING_CACHE_VERSION and metadata.get("positive_rows") == count
        except (OSError, json.JSONDecodeError):
            pass
    if cache_hit:
        return np.memmap(data_path, dtype=TRAINING_DTYPE, mode="r", shape=(count,)), True
    item_index = {item_id: index for index, item_id in enumerate(reference["item_ids"])}
    data = np.memmap(data_path, dtype=TRAINING_DTYPE, mode="w+", shape=(count,))
    position = 0
    for user_id, item_id, rating, timestamp in iter_rows(chunks):
        if rating < RATING_THRESHOLD:
            continue
        if classify_row(user_id, item_id, timestamp, reference["bounds"][user_id]) != "train":
            continue
        if position >= count:
            fail("training cache received more rows than the reference count")
        data[position] = (user_id, item_index[item_id], rating, timestamp)
        position += 1
    if position != count:
        fail(f"training cache row count mismatch: wrote {position}, expected {count}")
    data.flush()
    metadata_path.write_text(json.dumps({
        "cache_version": TRAINING_CACHE_VERSION,
        "cache_key": key,
        "positive_rows": count,
        "item_count": len(reference["item_ids"]),
        "dtype": str(TRAINING_DTYPE.descr),
    }, indent=2) + "\n", encoding="utf-8")
    return np.memmap(data_path, dtype=TRAINING_DTYPE, mode="r", shape=(count,)), False


def rank_orders(counts: np.ndarray, rating_sums: np.ndarray, item_ids: np.ndarray, prior: float, smoothing: float) -> dict[str, np.ndarray]:
    popularity = np.lexsort((item_ids, -counts))
    smoothed_mean = (rating_sums + smoothing * prior) / (counts + smoothing)
    weighted_score = np.where(counts > 0, counts * smoothed_mean, 0.0)
    weighted = np.lexsort((item_ids, -weighted_score))
    return {"popularity": popularity, "rating_weighted_popularity": weighted}


def metric_at_k(order: Iterable[int], item_ids: np.ndarray, seen: set[int], relevant: set[int], cutoff: int) -> tuple[float, float]:
    if not relevant:
        return 0.0, 0.0
    hits = 0
    dcg = 0.0
    selected = 0
    for index in order:
        item_id = int(item_ids[int(index)])
        if item_id in seen:
            continue
        selected += 1
        if item_id in relevant:
            hits += 1
            dcg += 1.0 / math.log2(selected + 1.0)
        if selected >= cutoff:
            break
    ideal_hits = min(cutoff, len(relevant))
    ideal_dcg = sum(1.0 / math.log2(rank + 1.0) for rank in range(1, ideal_hits + 1))
    return hits / len(relevant), dcg / ideal_dcg if ideal_dcg else 0.0


def user_metrics(order: Iterable[int], item_ids: np.ndarray, records: dict[str, set[int]], cutoff: int) -> tuple[float, float]:
    return metric_at_k(order, item_ids, records["train"], records["test"], cutoff)


def build_item_item_scores(
    training: np.memmap,
    selected: np.ndarray,
    user_index: np.ndarray,
    panel_user_indices: np.ndarray,
    support_indices: np.ndarray,
    item_count: int,
    user_count: int,
    score_limit: int,
    batch_size: int = 200,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Compute panel item-item scores without materializing an item-item matrix."""
    empty = (np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32))
    if not len(support_indices):
        return [empty for _ in panel_user_indices]
    support_local = np.full(item_count, -1, dtype=np.int32)
    support_local[support_indices] = np.arange(len(support_indices), dtype=np.int32)
    selected_rows = np.flatnonzero(selected)
    local = support_local[training["item_index"][selected_rows]]
    valid = local >= 0
    if not np.any(valid):
        return [empty for _ in panel_user_indices]
    matrix = sparse.csr_matrix(
        (
            np.ones(int(valid.sum()), dtype=np.float32),
            (user_index[selected_rows[valid]], local[valid]),
        ),
        shape=(user_count, len(support_indices)),
        dtype=np.float32,
    )
    matrix.data[:] = 1.0
    document_frequency = np.asarray(matrix.sum(axis=0)).ravel()
    inverse_norm = 1.0 / np.sqrt(np.maximum(document_frequency, 1.0))
    normalized = (matrix @ sparse.diags(inverse_norm)).tocsr()
    profile_local = [
        normalized.getrow(int(user_index_value)).indices
        for user_index_value in panel_user_indices
    ]
    output: list[tuple[np.ndarray, np.ndarray]] = []
    for start in range(0, len(panel_user_indices), batch_size):
        batch_indices = panel_user_indices[start : start + batch_size]
        overlap = (normalized[batch_indices] @ normalized.T).tocsr()
        batch_scores = (overlap @ normalized).tocsr()
        for row_index in range(batch_scores.shape[0]):
            row_start = batch_scores.indptr[row_index]
            row_end = batch_scores.indptr[row_index + 1]
            local_indices = batch_scores.indices[row_start:row_end]
            values = batch_scores.data[row_start:row_end]
            profile = profile_local[start + row_index]
            keep_profile = ~np.isin(local_indices, profile)
            local_indices = local_indices[keep_profile]
            values = values[keep_profile]
            if len(values) > score_limit:
                keep = np.argpartition(-values, score_limit - 1)[:score_limit]
                local_indices = local_indices[keep]
                values = values[keep]
            output.append(
                (
                    support_indices[local_indices].astype(np.int32, copy=False),
                    values.astype(np.float32, copy=False),
                )
            )
    return output


def chain(first: Iterable[int], second: Iterable[int]) -> Iterator[int]:
    yield from first
    yield from second


def item_item_rank_order(
    score_row: tuple[np.ndarray, np.ndarray],
    item_ids: np.ndarray,
) -> np.ndarray:
    indices, values = score_row
    order = np.lexsort((item_ids[indices], -values))
    return indices[order]


def item_item_user_metrics(
    ranked: np.ndarray,
    item_ids: np.ndarray,
    records: dict[str, set[int]],
    cutoff: int,
) -> tuple[float, float]:
    ranked_set = set(int(index) for index in ranked)
    fallback = (index for index in range(len(item_ids)) if index not in ranked_set)
    return user_metrics(chain((int(index) for index in ranked), fallback), item_ids, records, cutoff)


def reference_user_metrics(
    reference: dict[str, object],
    item_ids: np.ndarray,
    reference_item_scores: list[tuple[np.ndarray, np.ndarray]],
    cutoff: int,
    smoothing: float,
    models: tuple[str, ...],
) -> dict[str, dict[int, tuple[float, float]]]:
    counts = np.array(
        [reference["full_item_counts"].get(int(item), 0) for item in item_ids],
        dtype=np.float64,
    )
    sums = np.array(
        [reference["full_item_rating_sums"].get(int(item), 0.0) for item in item_ids],
        dtype=np.float64,
    )
    orders = rank_orders(counts, sums, item_ids, float(reference["positive_mean"]), smoothing)
    ranked_item_scores = [
        item_item_rank_order(row, item_ids) for row in reference_item_scores
    ]
    output = {model: {} for model in models}
    for panel_index, user_id in enumerate(reference["panel"]):
        records = reference["panel_records"][user_id]
        for model in models:
            if model == "item_item_cosine":
                value = item_item_user_metrics(
                    ranked_item_scores[panel_index],
                    item_ids,
                    records,
                    cutoff,
                )
            else:
                value = user_metrics(orders[model], item_ids, records, cutoff)
            output[model][user_id] = value
    return output



def sampling_decisions(scheme: str, training: np.memmap, reference: dict[str, object], user_index: np.ndarray, seed: int) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    rng = np.random.default_rng(seed)
    if scheme in ("uniform_user", "activity_stratified_user"):
        draws = rng.random(len(reference["user_ids"]))
        if scheme == "activity_stratified_user":
            strata = np.array([reference["activity_strata"].get(u, 0) for u in reference["user_ids"]], dtype=np.int8)
            for level in range(10):
                indexes = np.flatnonzero(strata == level)
                draws[indexes] = rng.random(len(indexes))
        return draws[user_index], None, None
    if scheme == "uniform_interaction":
        return rng.random(len(training)), None, None
    if scheme == "within_user_history":
        random_values = rng.random(len(training))
        order = np.lexsort((random_values, training["user_id"]))
        ordered_users = training["user_id"][order]
        _, starts, counts = np.unique(ordered_users, return_index=True, return_counts=True)
        ranks_sorted = np.arange(len(training), dtype=np.int64) - np.repeat(starts, counts)
        ranks = np.empty(len(training), dtype=np.int64)
        ranks[order] = ranks_sorted
        counts_by_user = np.bincount(user_index, minlength=len(reference["user_ids"]))
        return None, ranks, counts_by_user[user_index]
    fail(f"unknown sampling scheme: {scheme}")


def kendall_tau(sample_order: tuple[str, ...], reference_order: tuple[str, ...]) -> float:
    positions = {name: index for index, name in enumerate(reference_order)}
    sequence = [positions[name] for name in sample_order]
    concordant = discordant = 0
    for left in range(len(sequence)):
        for right in range(left + 1, len(sequence)):
            if sequence[left] < sequence[right]:
                concordant += 1
            elif sequence[left] > sequence[right]:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else 1.0


def run_scheme_replicate(
    scheme: str,
    replicate: int,
    fractions: tuple[float, ...],
    training: np.memmap,
    reference: dict[str, object],
    item_ids: np.ndarray,
    user_index: np.ndarray,
    panel_user_indices: np.ndarray,
    support_indices: np.ndarray,
    score_limit: int,
    reference_item_scores: list[tuple[np.ndarray, np.ndarray]],
    reference_metrics: dict[str, dict[int, tuple[float, float]]],
    cutoff: int,
    smoothing: float,
    models: tuple[str, ...],
    base_seed: int,
) -> list[dict[str, object]]:
    started = time.perf_counter()
    seed = stable_seed("sample_generalization_v2", scheme, replicate) ^ base_seed
    train_users = training["user_id"]
    train_items = training["item_index"]
    panel = reference["panel"]
    panel_mask = np.isin(train_users, np.asarray(panel, dtype=np.int32))
    value_draw, ranks, user_counts = sampling_decisions(scheme, training, reference, user_index, seed)
    rows: list[dict[str, object]] = []
    for fraction in fractions:
        if fraction == 1.0 and replicate > 0:
            continue
        fraction_started = time.perf_counter()
        if ranks is None:
            selected = value_draw < fraction
        else:
            selected = ranks < np.ceil(fraction * user_counts).astype(np.int64)
        selected_items = train_items[selected]
        counts = np.bincount(selected_items, minlength=len(item_ids)).astype(np.float64)
        rating_sums = np.bincount(
            selected_items,
            weights=training["rating"][selected].astype(np.float64),
            minlength=len(item_ids),
        )
        sample_interactions = int(selected.sum())
        population_fraction = sample_interactions / max(1, int(reference["positive_rows"]))
        sample_records = {u: set() for u in panel}
        selected_panel = selected & panel_mask
        for user_id, item_index_value in zip(train_users[selected_panel], train_items[selected_panel]):
            sample_records[int(user_id)].add(int(item_ids[int(item_index_value)]))
        supported = sum(bool(sample_records[u]) for u in panel)
        orders = rank_orders(
            counts,
            rating_sums,
            item_ids,
            float(reference["positive_mean"]),
            smoothing,
        )
        if "item_item_cosine" in models:
            if fraction == 1.0:
                sample_item_scores = reference_item_scores
            else:
                sample_item_scores = build_item_item_scores(
                    training,
                    selected,
                    user_index,
                    panel_user_indices,
                    support_indices,
                    len(item_ids),
                    len(reference["user_ids"]),
                    score_limit,
                )
        else:
            sample_item_scores = []
        evaluated = [u for u in panel if reference["panel_records"][u]["test"]]
        panel_position = {user_id: index for index, user_id in enumerate(panel)}
        sample_by_model: dict[str, list[tuple[float, float]]] = {}
        native_by_model: dict[str, list[tuple[float, float]]] = {}
        reference_by_model: dict[str, list[tuple[float, float]]] = {}
        for model in models:
            sample_values: list[tuple[float, float]] = []
            native_values: list[tuple[float, float]] = []
            reference_values: list[tuple[float, float]] = []
            for user_id in evaluated:
                fixed = {
                    "train": reference["panel_records"][user_id]["train"],
                    "test": reference["panel_records"][user_id]["test"],
                }
                native = {
                    "train": sample_records[user_id],
                    "test": reference["panel_records"][user_id]["test"],
                }
                if model == "item_item_cosine":
                    ranked_scores = item_item_rank_order(
                        sample_item_scores[panel_position[user_id]],
                        item_ids,
                    )
                    sample_value = item_item_user_metrics(ranked_scores, item_ids, fixed, cutoff)
                    native_value = item_item_user_metrics(ranked_scores, item_ids, native, cutoff)
                else:
                    sample_value = user_metrics(orders[model], item_ids, fixed, cutoff)
                    native_value = user_metrics(orders[model], item_ids, native, cutoff)
                sample_values.append(sample_value)
                native_values.append(native_value)
                reference_values.append(reference_metrics[model][user_id])
            sample_by_model[model] = sample_values
            native_by_model[model] = native_values
            reference_by_model[model] = reference_values
        sample_means = {
            model: float(np.mean([value[1] for value in values])) if values else 0.0
            for model, values in sample_by_model.items()
        }
        reference_means = {
            model: float(np.mean([value[1] for value in values])) if values else 0.0
            for model, values in reference_by_model.items()
        }
        sample_order = tuple(sorted(models, key=lambda model: (-sample_means[model], model)))
        reference_order = tuple(sorted(models, key=lambda model: (-reference_means[model], model)))
        tau = kendall_tau(sample_order, reference_order)
        fpc = math.sqrt(max(0.0, 1.0 - population_fraction))
        for model in models:
            sample_values = sample_by_model[model]
            native_values = native_by_model[model]
            reference_values = reference_by_model[model]
            sample_ndcg = float(np.mean([value[1] for value in sample_values])) if sample_values else None
            native_ndcg = float(np.mean([value[1] for value in native_values])) if native_values else None
            reference_ndcg = float(np.mean([value[1] for value in reference_values])) if reference_values else None
            sample_recall = float(np.mean([value[0] for value in sample_values])) if sample_values else None
            native_recall = float(np.mean([value[0] for value in native_values])) if native_values else None
            reference_recall = float(np.mean([value[0] for value in reference_values])) if reference_values else None
            rows.append(
                {
                    "scheme": scheme,
                    "replicate": replicate,
                    "seed": int(seed),
                    "fraction": fraction,
                    "cell_role": "primary_independence" if fraction <= 0.10 else "convergence_check",
                    "model": model,
                    "sample_interactions": sample_interactions,
                    "population_fraction": population_fraction,
                    "finite_population_correction": fpc,
                    "sample_users": int(np.unique(train_users[selected]).size),
                    "sample_items": int(np.count_nonzero(counts)),
                    "panel_users": len(panel),
                    "evaluated_users": len(evaluated),
                    "sample_supported_users": supported,
                    "sample_support_rate": supported / len(evaluated) if evaluated else None,
                    "sample_recall_at_10": sample_recall,
                    "sample_native_recall_at_10": native_recall,
                    "reference_recall_at_10": reference_recall,
                    "sample_ndcg_at_10": sample_ndcg,
                    "sample_native_ndcg_at_10": native_ndcg,
                    "reference_ndcg_at_10": reference_ndcg,
                    "absolute_ndcg_error": None if sample_ndcg is None else abs(sample_ndcg - reference_ndcg),
                    "relative_ndcg_error": None if sample_ndcg is None or not reference_ndcg else (sample_ndcg - reference_ndcg) / abs(reference_ndcg),
                    "absolute_recall_error": None if sample_recall is None else abs(sample_recall - reference_recall),
                    "native_frame_ndcg_gap": None if sample_ndcg is None or native_ndcg is None else native_ndcg - sample_ndcg,
                    "sample_algorithm_order": list(sample_order),
                    "reference_algorithm_order": list(reference_order),
                    "algorithm_order_kendall_tau": tau,
                    "top_choice_agreement": sample_order[0] == reference_order[0],
                    "evaluation_seconds": time.perf_counter() - fraction_started,
                }
            )
        if fraction == 1.0:
            current_rows = rows[-len(models):]
            for row in current_rows:
                if int(row["sample_interactions"]) != int(reference["positive_rows"]):
                    fail(f"fraction-1.0 sample count mismatch for {scheme}/{row['model']}")
                if row["absolute_ndcg_error"] is None or float(row["absolute_ndcg_error"]) > 1e-12:
                    fail(f"fraction-1.0 metric invariant failed for {scheme}/{row['model']}")
    pass_seconds = time.perf_counter() - started
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss_mb = rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024
    for row in rows:
        row["replicate_pass_seconds"] = pass_seconds
        row["amortized_seconds_per_fraction"] = pass_seconds / len(fractions)
        row["peak_rss_mb"] = peak_rss_mb
    return rows


def aggregate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, float, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["scheme"]), float(row["fraction"]), str(row["model"])), []).append(row)
    output = []
    for (scheme, fraction, model), values in sorted(groups.items()):
        def mean(field: str) -> float | None:
            numbers = [float(v[field]) for v in values if v[field] is not None]
            return float(np.mean(numbers)) if numbers else None
        def std(field: str) -> float | None:
            numbers = [float(v[field]) for v in values if v[field] is not None]
            return float(np.std(numbers, ddof=1)) if len(numbers) > 1 else (0.0 if numbers else None)
        sample_std = std("sample_ndcg_at_10")
        output.append({
            "scheme": scheme, "fraction": fraction, "cell_role": values[0]["cell_role"], "model": model,
            "replicate_count": len(values), "mean_sample_ndcg_at_10": mean("sample_ndcg_at_10"),
            "std_sample_ndcg_at_10": sample_std, "fpc_adjusted_std_sample_ndcg_at_10": None if sample_std is None else sample_std * (mean("finite_population_correction") or 0.0),
            "mean_reference_ndcg_at_10": mean("reference_ndcg_at_10"), "mean_absolute_ndcg_error": mean("absolute_ndcg_error"),
            "mean_relative_ndcg_error": mean("relative_ndcg_error"), "mean_sample_native_ndcg_at_10": mean("sample_native_ndcg_at_10"),
            "mean_absolute_native_frame_gap": mean("native_frame_ndcg_gap"), "mean_sample_recall_at_10": mean("sample_recall_at_10"),
            "mean_reference_recall_at_10": mean("reference_recall_at_10"), "mean_absolute_recall_error": mean("absolute_recall_error"),
            "mean_evaluated_users": mean("evaluated_users"), "mean_sample_supported_users": mean("sample_supported_users"),
            "mean_sample_support_rate": mean("sample_support_rate"), "mean_sample_users": mean("sample_users"),
            "mean_sample_items": mean("sample_items"), "mean_sample_interactions": mean("sample_interactions"),
            "mean_population_fraction": mean("population_fraction"), "mean_finite_population_correction": mean("finite_population_correction"),
            "mean_evaluation_seconds": mean("evaluation_seconds"), "mean_replicate_pass_seconds": mean("replicate_pass_seconds"),
            "mean_peak_rss_mb": mean("peak_rss_mb"), "mean_algorithm_order_kendall_tau": mean("algorithm_order_kendall_tau"),
            "top_choice_agreement_rate": mean("top_choice_agreement"),
        })
    return output


def save_figures(output_dir: Path, aggregates: list[dict[str, object]], models: tuple[str, ...]) -> list[str]:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    schemes = sorted({str(row["scheme"]) for row in aggregates})
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(1, len(schemes))))
    scheme_colors = dict(zip(schemes, colors))
    paths: list[str] = []
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 4), squeeze=False)
    for index, model in enumerate(models):
        axis = axes[0][index]
        for scheme in schemes:
            values = sorted((v for v in aggregates if v["scheme"] == scheme and v["model"] == model), key=lambda v: float(v["fraction"]))
            axis.errorbar([v["fraction"] for v in values], [v["mean_sample_ndcg_at_10"] for v in values], yerr=[v["std_sample_ndcg_at_10"] for v in values], marker="o", label=scheme, color=scheme_colors[scheme])
        axis.set_xscale("log"); axis.set_xlabel("Training interaction fraction"); axis.set_ylabel("NDCG@10"); axis.set_title(model.replace("_", " ")); axis.grid(alpha=0.2)
    axes[0][0].legend(frameon=False, fontsize=8); fig.tight_layout()
    path = figure_dir / "metric_vs_fraction.png"; fig.savefig(path, dpi=150); plt.close(fig); paths.append(str(path.relative_to(output_dir)))

    fig, axis = plt.subplots(figsize=(8, 4))
    for scheme in schemes:
        for index, model in enumerate(models):
            values = sorted((v for v in aggregates if v["scheme"] == scheme and v["model"] == model), key=lambda v: float(v["fraction"]))
            axis.plot([v["fraction"] for v in values], [v["mean_absolute_ndcg_error"] for v in values], marker="o", label=f"{scheme} / {model}", color=scheme_colors[scheme], linestyle=("-", "--", ":")[min(index, 2)])
    axis.set_xscale("log"); axis.set_xlabel("Training interaction fraction"); axis.set_ylabel("Absolute NDCG@10 error"); axis.grid(alpha=0.2); axis.legend(frameon=False, fontsize=7, ncol=2); fig.tight_layout()
    path = figure_dir / "relative_error_vs_fraction.png"; fig.savefig(path, dpi=150); plt.close(fig); paths.append(str(path.relative_to(output_dir)))

    fig, axis = plt.subplots(figsize=(8, 4))
    for scheme in schemes:
        values = sorted((v for v in aggregates if v["scheme"] == scheme and v["model"] == models[0]), key=lambda v: float(v["fraction"]))
        axis.plot([v["fraction"] for v in values], [v["mean_sample_users"] for v in values], marker="o", label=f"{scheme}: users", color=scheme_colors[scheme])
        axis.plot([v["fraction"] for v in values], [v["mean_sample_items"] for v in values], marker="x", linestyle=":", label=f"{scheme}: items", color=scheme_colors[scheme])
    axis.set_xscale("log"); axis.set_yscale("log"); axis.set_xlabel("Training interaction fraction"); axis.set_ylabel("Realized sampled entities"); axis.grid(alpha=0.2); axis.legend(frameon=False, fontsize=7, ncol=2); fig.tight_layout()
    path = figure_dir / "coverage_vs_fraction.png"; fig.savefig(path, dpi=150); plt.close(fig); paths.append(str(path.relative_to(output_dir)))

    fig, axis = plt.subplots(figsize=(8, 4))
    for scheme in schemes:
        values = sorted((v for v in aggregates if v["scheme"] == scheme and v["model"] == models[0]), key=lambda v: float(v["fraction"]))
        axis.plot([v["fraction"] for v in values], [v["top_choice_agreement_rate"] for v in values], marker="o", label=f"{scheme}: top choice", color=scheme_colors[scheme])
        axis.plot([v["fraction"] for v in values], [v["mean_algorithm_order_kendall_tau"] for v in values], marker="x", linestyle=":", label=f"{scheme}: Kendall tau", color=scheme_colors[scheme])
    axis.set_xscale("log"); axis.set_ylim(-1.05, 1.05); axis.set_xlabel("Training interaction fraction"); axis.set_ylabel("Agreement with full-data order"); axis.grid(alpha=0.2); axis.legend(frameon=False, fontsize=7, ncol=2); fig.tight_layout()
    path = figure_dir / "algorithm_order_agreement.png"; fig.savefig(path, dpi=150); plt.close(fig); paths.append(str(path.relative_to(output_dir)))

    fig, axis = plt.subplots(figsize=(8, 4))
    for scheme in schemes:
        values = sorted((v for v in aggregates if v["scheme"] == scheme and v["model"] == models[0]), key=lambda v: float(v["fraction"]))
        axis.plot([v["fraction"] for v in values], [v["mean_absolute_native_frame_gap"] for v in values], marker="o", label=scheme, color=scheme_colors[scheme])
    axis.set_xscale("log"); axis.set_xlabel("Training interaction fraction"); axis.set_ylabel("Native-frame minus fixed-frame NDCG@10"); axis.grid(alpha=0.2); axis.legend(frameon=False, fontsize=8); fig.tight_layout()
    path = figure_dir / "native_frame_gap_vs_fraction.png"; fig.savefig(path, dpi=150); plt.close(fig); paths.append(str(path.relative_to(output_dir)))

    fig, axis = plt.subplots(figsize=(8, 4))
    for scheme in schemes:
        values = sorted((v for v in aggregates if v["scheme"] == scheme and v["model"] == models[0]), key=lambda v: float(v["fraction"]))
        axis.plot([v["mean_sample_interactions"] for v in values], [v["mean_absolute_ndcg_error"] for v in values], marker="o", label=scheme, color=scheme_colors[scheme])
    axis.set_xscale("log"); axis.set_xlabel("Realized sampled training interactions"); axis.set_ylabel("Absolute NDCG@10 error"); axis.grid(alpha=0.2); axis.legend(frameon=False, fontsize=8); fig.tight_layout()
    path = figure_dir / "cost_vs_error.png"; fig.savefig(path, dpi=150); plt.close(fig); paths.append(str(path.relative_to(output_dir)))
    return paths


def fmt(value: object, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def write_report(output_dir: Path, reference: dict[str, object], rows: list[dict[str, object]], aggregates: list[dict[str, object]], figures: list[str], schemes: tuple[str, ...], fractions: tuple[float, ...], replicates: int, models: tuple[str, ...], support_threshold: int) -> None:
    lines = [
        "# Sample-generalization local run", "", "Status: exploratory full-snapshot evidence; no external-data claim.", "",
        "## Design", "",
        "- The full-data reference is computed once, persisted in `reference_artifact.json`, and reused for every candidate.",
        f"- MovieLens `ml-latest` 2023-07-20: {reference['raw_rows']:,} raw ratings, {len(reference['user_ids']):,} users, {len(reference['item_ids']):,} items.",
        f"- Positive training interactions: {reference['positive_rows']:,}; fixed evaluation panel: {len(reference['panel']):,} users.",
        f"- Schemes: {', '.join(schemes)}; models: {', '.join(models)}.",
        f"- Fractions: {', '.join(f'{x:g}' for x in fractions)}; draws per cell: {replicates}.",
        "- Fractions <= 10% are the primary near-independent ladder; larger fractions are convergence checks with finite-population correction.",
        "- Headline metrics use full-reference training-positive exclusion. Sample-native exclusion is a separate sensitivity frame.",
        f"- Item-item cosine uses fixed full-reference support >= {support_threshold}; unsupported candidates receive zero score rather than being removed.",
        "", "## Aggregate evidence", "",
        "| Scheme | Fraction | Role | Model | Mean NDCG | Abs. error | Native-fixed gap | Support | Kendall tau | Top choice | Sample rows |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for value in aggregates:
        lines.append(f"| {value['scheme']} | {float(value['fraction']):g} | {value['cell_role']} | {value['model']} | {fmt(value['mean_sample_ndcg_at_10'])} | {fmt(value['mean_absolute_ndcg_error'])} | {fmt(value['mean_absolute_native_frame_gap'])} | {fmt(value['mean_sample_supported_users'], 1)} / {fmt(value['mean_evaluated_users'], 1)} | {fmt(value['mean_algorithm_order_kendall_tau'], 3)} | {fmt(value['top_choice_agreement_rate'], 3)} | {fmt(value['mean_sample_interactions'], 0)} |")
    evaluation_seconds = [float(row["evaluation_seconds"]) for row in rows]
    replicate_seconds = [float(row["replicate_pass_seconds"]) for row in rows]
    peak_rss = [float(row["peak_rss_mb"]) for row in rows]
    lines += [
        "",
        "## Resource evidence",
        "",
        f"- Reference build: {float(reference['reference_seconds']):.1f}s; reference cache hit: {bool(reference['reference_cache_hit'])}.",
        f"- Per-cell evaluation time: {min(evaluation_seconds):.2f}–{max(evaluation_seconds):.2f}s; replicate pass time: {min(replicate_seconds):.1f}–{max(replicate_seconds):.1f}s.",
        f"- Peak resident set size across draw rows: {min(peak_rss):.0f}–{max(peak_rss):.0f} MiB.",
        "- These resource figures describe this local run and hardware; they are not a cross-machine performance claim.",
        "",
        "## Interpretation",
        "",
        "- Fixed-frame error is an in-reference approximation comparison, not an independent external-generalization test.",
        "- Native/fixed divergence measures exclusion-set protocol bias.",
        "- Support is essential for per-user models; it is diagnostic rather than a failure condition for global item-statistic controls.",
        "- A positive approximation claim requires metric tolerance, stable ordering, coverage/temporal checks, and replication by an independent scheme.",
        "- If the full reference is unaffordable, rerun with a declared largest-affordable reference (for example 50k users); conclusions then describe convergence to that reference and extrapolation beyond it.",
        "",
        "## Figures",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    ratings_dir, manifest_path = repo_path(args.ratings_dir), repo_path(args.manifest)
    output_dir, cache_dir = repo_path(args.output), repo_path(args.cache_dir)
    fractions = tuple(sorted({float(x) for x in args.fractions.split(",") if x.strip()}))
    schemes = tuple(x.strip() for x in args.schemes.split(",") if x.strip())
    models = tuple(x.strip() for x in args.models.split(",") if x.strip())
    if not fractions or fractions[0] <= 0 or fractions[-1] > 1: fail("fractions must be in (0, 1]")
    if args.replicates <= 0 or args.panel_size <= 0: fail("replicates and panel size must be positive")
    if set(schemes) - set(PRIMARY_SCHEMES): fail(f"unknown schemes: {sorted(set(schemes) - set(PRIMARY_SCHEMES))}")
    if not models or set(models) - set(MODELS): fail(f"models must be selected from {MODELS}")
    if args.support_threshold <= 0:
        fail("support threshold must be positive")
    chunks = rating_chunks(ratings_dir)
    if not manifest_path.is_file(): fail(f"missing input file: {manifest_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_sha = sha256_file(manifest_path)
    print("building or loading full-data reference", flush=True)
    reference, cache_key = load_or_build_reference(chunks, manifest_sha, args.panel_size, args.seed, cache_dir)
    print("building or loading positive-training cache", flush=True)
    training, training_cache_hit = load_training_cache(chunks, reference, cache_dir, cache_key)
    item_ids = np.asarray(reference["item_ids"], dtype=np.int64)
    user_ids = np.asarray(reference["user_ids"], dtype=np.int32)
    user_index = np.searchsorted(user_ids, training["user_id"]).astype(np.int32)
    panel_user_indices = np.searchsorted(
        user_ids,
        np.asarray(reference["panel"], dtype=np.int32),
    ).astype(np.int32)
    support_indices = np.asarray(
        [
            i
            for i, item in enumerate(item_ids)
            if reference["full_item_counts"].get(int(item), 0) >= args.support_threshold
        ],
        dtype=np.int32,
    )
    max_fixed_exclusion = max(
        (len(reference["panel_records"][user_id]["train"]) for user_id in reference["panel"]),
        default=0,
    )
    score_limit = max(args.item_item_top_k, args.cutoff + max_fixed_exclusion)
    print("building full-data model reference", flush=True)
    if "item_item_cosine" in models:
        reference_item_scores = build_item_item_scores(
            training,
            np.ones(len(training), dtype=np.bool_),
            user_index,
            panel_user_indices,
            support_indices,
            len(item_ids),
            len(user_ids),
            score_limit,
        )
    else:
        reference_item_scores = [
            (np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32))
            for _ in reference["panel"]
        ]
    reference_metrics = reference_user_metrics(
        reference,
        item_ids,
        reference_item_scores,
        args.cutoff,
        args.smoothing_count,
        models,
    )
    panel_metrics = {
        model: {
            str(user): [float(value[0]), float(value[1])]
            for user, value in values.items()
        }
        for model, values in reference_metrics.items()
    }
    (output_dir / "reference_artifact.json").write_text(json.dumps({
        "reference_id": cache_key,
        "scope": "full_snapshot_once",
        "reuse_rule": "build once, cache, and diff every candidate against this frozen reference",
        "manifest_path": str(manifest_path.relative_to(ROOT)),
        "manifest_sha256": manifest_sha,
        "dataset": {
            "raw_rows": reference["raw_rows"],
            "user_count": len(reference["user_ids"]),
            "item_count": len(reference["item_ids"]),
            "positive_training_rows": reference["positive_rows"],
            "eligible_user_count": len(reference["eligible_users"]),
            "panel_user_count": len(reference["panel"]),
        },
        "controls": {
            "positive_rule": "rating >= 4.0",
            "split": "chronological 80/10/10",
            "candidate_catalog": "all full-snapshot items; item-item unsupported candidates get zero score",
            "item_item_support_threshold": args.support_threshold,
            "item_item_top_k": args.item_item_top_k,
            "item_item_score_limit": score_limit,
        },
        "reference_build_seconds": reference["reference_seconds"],
        "reference_cache_hit": reference["reference_cache_hit"],
        "models": list(models),
        "panel_metrics": panel_metrics,
    }, indent=2) + "\n", encoding="utf-8")
    rows: list[dict[str, object]] = []
    for scheme_index, scheme in enumerate(schemes, 1):
        for replicate in range(args.replicates):
            print(
                f"sampling {scheme} replicate {replicate + 1}/{args.replicates} "
                f"({scheme_index}/{len(schemes)})",
                flush=True,
            )
            rows.extend(
                run_scheme_replicate(
                    scheme,
                    replicate,
                    fractions,
                    training,
                    reference,
                    item_ids,
                    user_index,
                    panel_user_indices,
                    support_indices,
                    score_limit,
                    reference_item_scores,
                    reference_metrics,
                    args.cutoff,
                    args.smoothing_count,
                    models,
                    args.seed,
                )
            )
    if 1.0 in fractions:
        full_rows = [row for row in rows if float(row["fraction"]) == 1.0]
        expected_full_rows = len(schemes) * len(models)
        if len(full_rows) != expected_full_rows:
            fail(f"fraction-1.0 invariant row count mismatch: {len(full_rows)} != {expected_full_rows}")
        for row in full_rows:
            if int(row["sample_interactions"]) != int(reference["positive_rows"]):
                fail(f"fraction-1.0 sample count mismatch for {row['scheme']}/{row['model']}")
            if row["absolute_ndcg_error"] is None or float(row["absolute_ndcg_error"]) > 1e-12:
                fail(f"fraction-1.0 metric invariant failed for {row['scheme']}/{row['model']}")
    aggregates = aggregate_rows(rows)
    figures = save_figures(output_dir, aggregates, models)
    summary = {
        "run_id": "sample_generalization_v2_local", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": {"path": str(Path(__file__).relative_to(ROOT)), "sha256": sha256_file(Path(__file__).resolve())},
        "inputs": {"manifest_path": str(manifest_path.relative_to(ROOT)), "manifest_sha256": manifest_sha, "ratings_chunks": [str(p.relative_to(ROOT)) for p in chunks], "ratings_chunk_count": len(chunks), "reference_cache_key": cache_key, "training_cache_hit": training_cache_hit},
        "design": {"schemes": list(schemes), "fractions": list(fractions), "replicates": args.replicates, "panel_size": args.panel_size, "cutoff": args.cutoff, "models": list(models), "smoothing_count": args.smoothing_count, "support_threshold": args.support_threshold, "item_item_top_k": args.item_item_top_k, "item_item_score_limit": score_limit, "positive_rule": "rating >= 4.0", "split": "chronological 80/10/10 over full snapshot", "reference_scope": "full_snapshot_once; persisted and reused", "evaluation": "frozen full-data test target over panel users with test positives; full-reference training positives excluded; sampled exclusion retained as sensitivity", "primary_independence_ladder": "fractions <= 0.10; larger fractions are convergence checks with finite-population correction"},
        "dataset": {"raw_rows": reference["raw_rows"], "user_count": len(reference["user_ids"]), "item_count": len(reference["item_ids"]), "positive_training_rows": reference["positive_rows"], "eligible_user_count": len(reference["eligible_users"]), "panel_user_count": len(reference["panel"]), "reference_build_seconds": reference["reference_seconds"], "reference_cache_hit": reference["reference_cache_hit"], "rating_counts": {str(k): v for k, v in sorted(reference["rating_counts"].items())}, "item_item_support_count": len(support_indices)},
        "reference_artifact": "reference_artifact.json", "rows": rows, "aggregates": aggregates, "figures": figures,
    }
    (output_dir / "candidate_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(output_dir, reference, rows, aggregates, figures, schemes, fractions, args.replicates, models, args.support_threshold)
    print(f"created {output_dir / 'candidate_summary.json'}", flush=True)
    print(f"created {output_dir / 'report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
