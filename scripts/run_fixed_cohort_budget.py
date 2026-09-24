#!/usr/bin/env python3
"""Run a fixed-cohort chronological local-data-budget replay on MovieLens.

This is an offline, policy-specific sample-to-reference experiment. It does not
simulate federated optimization, device availability, or a privacy mechanism.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor

# The ALS solves are batches of 32x32 systems. Multi-threaded BLAS spends more
# time launching threads than solving them: capping to one thread measured 7.7x
# faster per iteration on this machine. Must run before numpy is imported.
for _blas_threads in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_blas_threads, "1")

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import sparse
from run_sample_generalization import (
    RATING_THRESHOLD,
    chain,
    iter_rows,
    iter_user_histories,
    keyed_u64,
    metric_at_k,
    rank_orders,
    rating_chunks,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]
SECONDS_PER_DAY = 86_400
POLICIES = ("equal_chronological_cap", "tail_reserve_cap")
MODELS = (
    "popularity",
    "rating_weighted_popularity",
    "item_item_cosine",
    "implicit_als",
)
DETERMINISTIC_MODELS = ("popularity", "rating_weighted_popularity", "item_item_cosine")


@dataclass(frozen=True)
class Event:
    item_id: int
    rating: float
    timestamp: int


@dataclass
class Client:
    user_id: int
    pilot: list[Event]
    collection: list[Event]
    seen: set[int]
    test: set[int]
    test_before_seen_exclusion: int


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read {path}: {exc}")
    if not isinstance(payload, dict):
        fail(f"{path} must contain a JSON object")
    return payload


def nested(payload: dict[str, object], key: str) -> dict[str, object]:
    value = payload.get(key)
    if not isinstance(value, dict):
        fail(f"config field {key!r} must be an object")
    return value


def ordered_event_key(user_id: int, event: Event, seed: int) -> tuple[int, int]:
    return event.timestamp, keyed_u64(seed, user_id, event.item_id, event.timestamp)


def global_cutoffs(
    chunks: list[Path], pilot_fraction: float, collection_fraction: float
) -> tuple[int, int, dict[str, int], set[int]]:
    """Find day-aligned global cutoffs using positive ratings only."""
    if not 0 < pilot_fraction < 1 or not 0 < collection_fraction < 1:
        fail("window fractions must be strictly between zero and one")
    if pilot_fraction + collection_fraction >= 1:
        fail("pilot_fraction + collection_fraction must be less than one")

    day_counts: Counter[int] = Counter()
    catalogue: set[int] = set()
    raw_rows = 0
    positive_rows = 0
    for _, item_id, rating, timestamp in iter_rows(chunks):
        raw_rows += 1
        catalogue.add(item_id)
        if rating >= RATING_THRESHOLD:
            day_counts[timestamp // SECONDS_PER_DAY] += 1
            positive_rows += 1
    if not positive_rows:
        fail("no positive ratings found")

    def cutoff_at(fraction: float) -> tuple[int, int]:
        target = math.ceil(positive_rows * fraction)
        cumulative = 0
        for day in sorted(day_counts):
            cumulative += day_counts[day]
            if cumulative >= target:
                return (day + 1) * SECONDS_PER_DAY - 1, cumulative
        raise AssertionError("positive-rating cutoff was not found")

    pilot_cutoff, pilot_rows = cutoff_at(pilot_fraction)
    collection_cutoff, through_collection_rows = cutoff_at(
        pilot_fraction + collection_fraction
    )
    if pilot_cutoff >= collection_cutoff:
        fail("day-aligned pilot and collection cutoffs collapsed")
    return (
        pilot_cutoff,
        collection_cutoff,
        {
            "raw_rows": raw_rows,
            "positive_rows": positive_rows,
            "pilot_positive_rows": pilot_rows,
            "collection_positive_rows": through_collection_rows - pilot_rows,
            "future_positive_rows": positive_rows - through_collection_rows,
        },
        catalogue,
    )


def select_cohort(
    chunks: list[Path], pilot_cutoff: int, minimum_pilot_events: int,
    recent_pilot_activity_days: int, panel_size: int, seed: int,
) -> tuple[list[int], int]:
    if recent_pilot_activity_days <= 0:
        fail("recent_pilot_activity_days must be positive")
    recent_start = pilot_cutoff - recent_pilot_activity_days * SECONDS_PER_DAY
    eligible: list[int] = []
    for user_id, history in iter_user_histories(chunks):
        pilot_positive = 0
        recently_active = False
        for _, _, rating, timestamp in history:
            if rating < RATING_THRESHOLD or timestamp > pilot_cutoff:
                continue
            pilot_positive += 1
            if timestamp > recent_start:
                recently_active = True
        if pilot_positive >= minimum_pilot_events and recently_active:
            eligible.append(user_id)
    if len(eligible) < panel_size:
        fail(
            f"only {len(eligible)} recent pilot-eligible users for requested panel "
            f"of {panel_size}"
        )
    cohort = sorted(eligible, key=lambda user_id: (keyed_u64(seed, user_id), user_id))[
        :panel_size
    ]
    return sorted(cohort), len(eligible)


def load_clients(
    chunks: list[Path], cohort: list[int], pilot_cutoff: int,
    collection_cutoff: int, seed: int,
) -> dict[int, Client]:
    wanted = set(cohort)
    clients: dict[int, Client] = {}
    for user_id, history in iter_user_histories(chunks):
        if user_id not in wanted:
            continue
        pilot: list[Event] = []
        collection: list[Event] = []
        seen: set[int] = set()
        test_before_seen: set[int] = set()
        for _, item_id, rating, timestamp in history:
            event = Event(item_id, rating, timestamp)
            if timestamp <= pilot_cutoff:
                seen.add(item_id)
                if rating >= RATING_THRESHOLD:
                    pilot.append(event)
            elif timestamp <= collection_cutoff:
                seen.add(item_id)
                if rating >= RATING_THRESHOLD:
                    collection.append(event)
            elif rating >= RATING_THRESHOLD:
                test_before_seen.add(item_id)
        pilot.sort(key=lambda event: ordered_event_key(user_id, event, seed))
        collection.sort(key=lambda event: ordered_event_key(user_id, event, seed))
        clients[user_id] = Client(
            user_id=user_id,
            pilot=pilot,
            collection=collection,
            seen=seen,
            test=test_before_seen - seen,
            test_before_seen_exclusion=len(test_before_seen),
        )
    missing = sorted(wanted - clients.keys())
    if missing:
        fail(f"cohort users missing from ratings: {missing[:5]}")
    return clients


def item_counts(events: Iterable[Event]) -> Counter[int]:
    return Counter(event.item_id for event in events)


def tail_count_ceiling(pilot_counts: Counter[int], quantile: float) -> int:
    if not 0 < quantile <= 1:
        fail("tail pilot_item_count_quantile must be in (0, 1]")
    values = sorted(pilot_counts.values())
    if not values:
        fail("pilot history has no positive-item counts")
    index = max(0, math.ceil(len(values) * quantile) - 1)
    return int(values[index])


def equal_cap(events: list[Event], cap: int) -> list[Event]:
    return events[:cap]


def tail_reserve_cap(
    events: list[Event], cap: int, pilot_counts: Counter[int], tail_ceiling: int,
    reserve_fraction: float,
) -> list[Event]:
    if not 0 <= reserve_fraction <= 1:
        fail("tail reserve_fraction must be in [0, 1]")
    reserve = min(cap, math.ceil(cap * reserve_fraction))
    selected_indexes: set[int] = set()
    for index, event in enumerate(events):
        if pilot_counts.get(event.item_id, 0) <= tail_ceiling:
            selected_indexes.add(index)
            if len(selected_indexes) >= reserve:
                break
    for index in range(len(events)):
        if len(selected_indexes) >= cap:
            break
        selected_indexes.add(index)
    return [event for index, event in enumerate(events) if index in selected_indexes]


def selected_events(
    policy: str, clients: dict[int, Client], cap: int,
    pilot_counts: Counter[int], tail_ceiling: int, reserve_fraction: float,
) -> dict[int, list[Event]]:
    if cap <= 0:
        fail("cap must be positive")
    output: dict[int, list[Event]] = {}
    for user_id, client in clients.items():
        if policy == "equal_chronological_cap":
            output[user_id] = equal_cap(client.collection, cap)
        elif policy == "tail_reserve_cap":
            output[user_id] = tail_reserve_cap(
                client.collection, cap, pilot_counts, tail_ceiling, reserve_fraction
            )
        else:
            fail(f"unknown policy: {policy}")
    return output


def training_statistics(
    clients: dict[int, Client], selected: dict[int, list[Event]], item_index: dict[int, int]
) -> tuple[np.ndarray, np.ndarray, int, float]:
    counts = np.zeros(len(item_index), dtype=np.float64)
    rating_sums = np.zeros(len(item_index), dtype=np.float64)
    interactions = 0
    total_rating = 0.0
    for client in clients.values():
        for event in client.pilot:
            index = item_index[event.item_id]
            counts[index] += 1.0
            rating_sums[index] += event.rating
            interactions += 1
            total_rating += event.rating
    for events in selected.values():
        for event in events:
            index = item_index[event.item_id]
            counts[index] += 1.0
            rating_sums[index] += event.rating
            interactions += 1
            total_rating += event.rating
    if not interactions:
        fail("training data is empty")
    return counts, rating_sums, interactions, total_rating / interactions


def training_matrix(
    clients: dict[int, Client], selected: dict[int, list[Event]],
    user_row: dict[int, int], item_index: dict[int, int],
) -> sparse.csr_matrix:
    rows: list[int] = []
    columns: list[int] = []
    for user_id, client in clients.items():
        row = user_row[user_id]
        for event in client.pilot:
            rows.append(row)
            columns.append(item_index[event.item_id])
        for event in selected[user_id]:
            rows.append(row)
            columns.append(item_index[event.item_id])
    matrix = sparse.coo_matrix(
        (np.ones(len(rows), dtype=np.float64), (rows, columns)),
        shape=(len(user_row), len(item_index)),
    ).tocsr()
    matrix.sum_duplicates()
    return matrix


def als_factor_step(
    matrix: sparse.csr_matrix, other_factors: np.ndarray,
    regularization: float, alpha: float, max_block: int = 4_000_000,
) -> np.ndarray:
    """One ALS half-step, batched by interaction count.

    Rows with the same number of interactions are stacked and solved together,
    which removes the per-entity Python loop. The arithmetic is unchanged; the
    result matches the scalar formulation to about 1e-15.
    """
    factor_count = other_factors.shape[1]
    output = np.zeros((matrix.shape[0], factor_count), dtype=np.float64)
    base = other_factors.T @ other_factors + regularization * np.eye(factor_count)
    indptr, indices, data = matrix.indptr, matrix.indices, matrix.data
    counts = np.diff(indptr)
    active = np.flatnonzero(counts)
    if not len(active):
        return output
    active = active[np.argsort(counts[active], kind="stable")]
    sizes = counts[active]
    bounds = np.concatenate(([0], np.flatnonzero(np.diff(sizes)) + 1, [len(active)]))
    for group_index in range(len(bounds) - 1):
        group = active[bounds[group_index] : bounds[group_index + 1]]
        width = int(counts[group[0]])
        block = max(1, max_block // max(width * factor_count, 1))
        for start in range(0, len(group), block):
            chunk = group[start : start + block]
            offsets = indptr[chunk][:, None] + np.arange(width)
            vectors = other_factors[indices[offsets]]
            weight = alpha * data[offsets]
            left = np.matmul((vectors * weight[:, :, None]).transpose(0, 2, 1), vectors) + base
            right = (vectors * (weight + 1.0)[:, :, None]).sum(axis=1)
            output[chunk] = np.linalg.solve(left, right)
    return output


def train_implicit_als(
    matrix: sparse.csr_matrix, factor_count: int, regularization: float,
    alpha: float, iterations: int, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if factor_count <= 0 or iterations <= 0:
        fail("ALS factor count and iterations must be positive")
    if regularization <= 0 or alpha <= 0:
        fail("ALS regularization and alpha must be positive")
    rng = np.random.default_rng(seed)
    item_factors = rng.normal(0.0, 0.01, (matrix.shape[1], factor_count))
    transposed = matrix.T.tocsr()
    user_factors = np.zeros((matrix.shape[0], factor_count), dtype=np.float64)
    for _ in range(iterations):
        user_factors = als_factor_step(matrix, item_factors, regularization, alpha)
        item_factors = als_factor_step(transposed, user_factors, regularization, alpha)
    return user_factors, item_factors


def top_unseen_from_scores(
    scores: np.ndarray, item_ids: np.ndarray, seen_mask: np.ndarray, limit: int
) -> np.ndarray:
    """Rank the highest-scoring unseen candidates, ties broken by item id."""
    masked = np.where(seen_mask, -np.inf, scores)
    finite = int(np.count_nonzero(np.isfinite(masked)))
    take = min(limit, finite)
    if take <= 0:
        return np.empty(0, dtype=np.int64)
    candidates = np.argpartition(-masked, take - 1)[:take]
    candidates = candidates[np.isfinite(masked[candidates])]
    return candidates[np.lexsort((item_ids[candidates], -masked[candidates]))]


def top_unseen_from_order(
    order: np.ndarray, item_ids: np.ndarray, seen: set[int], limit: int
) -> list[int]:
    """Walk a precomputed global ranking and keep the first unseen candidates."""
    picked: list[int] = []
    for index in order:
        position = int(index)
        if int(item_ids[position]) in seen:
            continue
        picked.append(position)
        if len(picked) >= limit:
            break
    return picked


def item_item_scores(
    matrix: sparse.csr_matrix, rows: list[int], support_indices: np.ndarray,
    seen_masks: list[np.ndarray], item_ids: np.ndarray, limit: int,
    block_size: int = 2000, batch_size: int = 512,
) -> list[np.ndarray]:
    """Deterministic item-item cosine ranking for the requested user rows.

    Scoring is `N (N^T N)` rather than `(N N^T) N`. Matrix multiplication is
    associative, so the scores are unchanged, but the intermediate is the
    item-item matrix instead of the user-user matrix, which removes the
    quadratic-in-users cost. Candidate items are processed in blocks so the
    similarity matrix is never materialized in full.
    """
    if not len(support_indices):
        return [np.empty(0, dtype=np.int64) for _ in rows]
    restricted = matrix[:, support_indices].tocsr()
    restricted.data[:] = 1.0
    frequency = np.asarray(restricted.sum(axis=0)).ravel()
    normalized = (restricted @ sparse.diags(1.0 / np.sqrt(np.maximum(frequency, 1.0)))).tocsr()
    transposed = normalized.T.tocsr()
    support_item_ids = item_ids[support_indices]
    profiles = normalized[rows].tocsr()
    collected_scores: list[np.ndarray] = []
    collected_local: list[np.ndarray] = []
    for block_start in range(0, len(support_indices), block_size):
        stop = min(block_start + block_size, len(support_indices))
        similarity_block = (transposed @ normalized[:, block_start:stop]).toarray()
        width = similarity_block.shape[1]
        take = min(limit, width)
        for batch_start in range(0, len(rows), batch_size):
            batch = slice(batch_start, min(batch_start + batch_size, len(rows)))
            block_scores = profiles[batch] @ similarity_block
            selection = np.argpartition(-block_scores, take - 1, axis=1)[:, :take]
            if block_start == 0:
                collected_scores.append(np.take_along_axis(block_scores, selection, axis=1))
                collected_local.append(selection + block_start)
            else:
                index = batch_start // batch_size
                collected_scores[index] = np.concatenate(
                    [collected_scores[index], np.take_along_axis(block_scores, selection, axis=1)],
                    axis=1,
                )
                collected_local[index] = np.concatenate(
                    [collected_local[index], selection + block_start], axis=1
                )
    output: list[np.ndarray] = []
    for batch_index, (scores_block, local_block) in enumerate(zip(collected_scores, collected_local)):
        for offset in range(scores_block.shape[0]):
            row_index = batch_index * batch_size + offset
            local = local_block[offset]
            values = scores_block[offset]
            positive = values > 0.0
            local, values = local[positive], values[positive]
            global_index = support_indices[local].astype(np.int64, copy=False)
            keep = ~seen_masks[row_index][global_index]
            local, values = local[keep], values[keep]
            if len(values) > limit:
                selection = np.argpartition(-values, limit - 1)[:limit]
                local, values = local[selection], values[selection]
            ordered = np.lexsort((support_item_ids[local], -values))
            output.append(support_indices[local[ordered]].astype(np.int64, copy=False))
    return output


def user_cutoff_metrics(
    ranked: Iterable[int], item_ids: np.ndarray, relevant: set[int],
    weights: dict[int, float], cutoffs: tuple[int, ...],
) -> dict[int, dict[str, float]]:
    """Unweighted and propensity-weighted metrics at every cutoff.

    Weighted values follow the self-normalized inverse-propensity family: each
    relevant item contributes its inverse exposure propensity, and the ideal
    ranking uses the same weights sorted descending, so a perfect ranking still
    scores 1.0.
    """
    if not relevant:
        fail("a user with no relevant items must not be evaluated")
    deepest = max(cutoffs)
    hit_ranks: list[int] = []
    hit_weights: list[float] = []
    for position, index in enumerate(ranked, start=1):
        if position > deepest:
            break
        item_id = int(item_ids[int(index)])
        if item_id in relevant:
            hit_ranks.append(position)
            hit_weights.append(weights[item_id])
    relevant_size = len(relevant)
    total_weight = sum(weights[item] for item in relevant)
    sorted_weights = sorted((weights[item] for item in relevant), reverse=True)
    output: dict[int, dict[str, float]] = {}
    for cutoff in cutoffs:
        hits = sum(1 for rank in hit_ranks if rank <= cutoff)
        dcg = sum(
            1.0 / math.log2(rank + 1.0) for rank in hit_ranks if rank <= cutoff
        )
        weighted_dcg = sum(
            weight / math.log2(rank + 1.0)
            for rank, weight in zip(hit_ranks, hit_weights)
            if rank <= cutoff
        )
        weighted_hits = sum(
            weight for rank, weight in zip(hit_ranks, hit_weights) if rank <= cutoff
        )
        ideal_depth = min(cutoff, relevant_size)
        ideal_dcg = sum(1.0 / math.log2(rank + 1.0) for rank in range(1, ideal_depth + 1))
        weighted_ideal = sum(
            sorted_weights[rank - 1] / math.log2(rank + 1.0)
            for rank in range(1, ideal_depth + 1)
        )
        output[cutoff] = {
            "ndcg": dcg / ideal_dcg if ideal_dcg else 0.0,
            "precision": hits / float(cutoff),
            "recall": hits / float(relevant_size),
            "weighted_ndcg": weighted_dcg / weighted_ideal if weighted_ideal else 0.0,
            "weighted_recall": weighted_hits / total_weight if total_weight else 0.0,
        }
    return output


def propensity_weights(
    counts: np.ndarray, item_ids: np.ndarray, propensity_config: dict[str, object]
) -> dict[int, float]:
    """Inverse exposure weights from an item-popularity power law (S33).

    The exposure model is an assumption, not a measurement: MovieLens cannot
    validate it, so every weighted metric is published beside its unweighted
    counterpart.
    """
    gamma = float(propensity_config["gamma"])
    floor = float(propensity_config["minimum_propensity"])
    if not 0.0 < floor <= 1.0:
        fail("minimum_propensity must be in (0, 1]")
    exponent = (gamma + 1.0) / 2.0
    raw = np.power(np.maximum(counts, 0.0), exponent)
    peak = float(raw.max())
    if peak <= 0.0:
        fail("propensity model needs at least one observed item")
    propensity = np.maximum(raw / peak, floor)
    return {int(item_ids[index]): 1.0 / propensity[index] for index in range(len(item_ids))}


def activity_strata(
    clients: dict[int, Client], user_ids: list[int], labels: list[str]
) -> dict[int, str]:
    """Assign users to pilot-activity terciles, fixed across every policy."""
    sizes = np.array([len(clients[user_id].pilot) for user_id in user_ids], dtype=np.float64)
    lower, upper = np.quantile(sizes, [1 / 3, 2 / 3])
    assignment: dict[int, str] = {}
    for user_id, size in zip(user_ids, sizes):
        if size <= lower:
            assignment[user_id] = labels[0]
        elif size <= upper:
            assignment[user_id] = labels[1]
        else:
            assignment[user_id] = labels[2]
    return assignment


def popularity_bands(
    counts: np.ndarray, item_ids: np.ndarray, labels: list[str]
) -> dict[int, str]:
    """Classify candidate items into head, mid, and tail popularity bands."""
    observed = counts[counts > 0]
    if not len(observed):
        fail("popularity bands need at least one observed item")
    tail_ceiling = float(np.quantile(observed, 0.2))
    head_floor = float(np.quantile(observed, 0.9))
    bands: dict[int, str] = {}
    for index in range(len(item_ids)):
        value = counts[index]
        if value >= head_floor:
            bands[int(item_ids[index])] = labels[0]
        elif value <= tail_ceiling:
            bands[int(item_ids[index])] = labels[2]
        else:
            bands[int(item_ids[index])] = labels[1]
    return bands


def evaluation_ceilings(
    clients: dict[int, Client], user_ids: list[int], cutoff: int
) -> dict[str, object]:
    """Report where Recall@K is structurally capped by the relevant-set size."""
    sizes = np.array([len(clients[user_id].test) for user_id in user_ids], dtype=np.float64)
    ceilings = np.minimum(sizes, float(cutoff)) / sizes
    capped = sizes > float(cutoff)
    return {
        "definition": (
            "Recall@K divides by the full relevant set, so a user with more than K "
            "future positives cannot exceed K/|relevant|. NDCG@K and precision@K "
            "are reported beside it; NDCG normalizes by min(K, |relevant|)."
        ),
        "cutoff": int(cutoff),
        "relevant_set_size": {
            "mean": float(sizes.mean()),
            "median": float(np.median(sizes)),
            "p90": float(np.quantile(sizes, 0.9)),
            "max": float(sizes.max()),
        },
        "users_with_relevant_above_cutoff": int(capped.sum()),
        "share_with_relevant_above_cutoff": float(capped.mean()),
        "mean_recall_ceiling": float(ceilings.mean()),
        "min_recall_ceiling": float(ceilings.min()),
    }


def evaluate_condition(
    orders: dict[str, np.ndarray], item_item_rankings: list[np.ndarray],
    als_runs: list[tuple[np.ndarray, np.ndarray]], user_ids: list[int],
    user_row: dict[int, int], item_ids: np.ndarray, seen_masks: list[np.ndarray],
    clients: dict[int, Client], weights: dict[int, float], cutoffs: tuple[int, ...],
    hit_bands: dict[int, str], band_labels: list[str],
) -> dict[str, object]:
    """Per-user metrics for every model at every cutoff, plus hit-band counts."""
    if not als_runs:
        fail("at least one ALS seed run is required")
    if len(item_item_rankings) != len(user_ids):
        fail("item-item rankings must align with the evaluated users")
    deepest = max(cutoffs)
    fields = ("ndcg", "precision", "recall", "weighted_ndcg", "weighted_recall")
    per_user = {
        model: {cutoff: {field: np.empty(len(user_ids)) for field in fields} for cutoff in cutoffs}
        for model in MODELS
    }
    seed_primary = np.empty((len(als_runs), len(user_ids)), dtype=np.float64)
    band_counts = {model: {label: 0 for label in band_labels} for model in MODELS}
    primary = max(cutoffs)
    for row, user_id in enumerate(user_ids):
        client = clients[user_id]
        rankings: dict[str, list[int] | np.ndarray] = {
            "popularity": top_unseen_from_order(orders["popularity"], item_ids, client.seen, deepest),
            "rating_weighted_popularity": top_unseen_from_order(
                orders["rating_weighted_popularity"], item_ids, client.seen, deepest
            ),
            "item_item_cosine": item_item_rankings[row],
        }
        seed_metrics: list[dict[int, dict[str, float]]] = []
        for user_factors, item_factors in als_runs:
            ranking = top_unseen_from_scores(
                item_factors @ user_factors[user_row[user_id]],
                item_ids,
                seen_masks[row],
                deepest,
            )
            seed_metrics.append(
                user_cutoff_metrics(ranking, item_ids, client.test, weights, cutoffs)
            )
        rankings["implicit_als"] = top_unseen_from_scores(
            als_runs[0][1] @ als_runs[0][0][user_row[user_id]], item_ids, seen_masks[row], deepest
        )
        for model in MODELS:
            if model == "implicit_als":
                for cutoff in cutoffs:
                    for field in fields:
                        per_user[model][cutoff][field][row] = float(
                            np.mean([run[cutoff][field] for run in seed_metrics])
                        )
                seed_primary[:, row] = [run[primary]["ndcg"] for run in seed_metrics]
            else:
                metrics = user_cutoff_metrics(
                    rankings[model], item_ids, client.test, weights, cutoffs
                )
                for cutoff in cutoffs:
                    for field in fields:
                        per_user[model][cutoff][field][row] = metrics[cutoff][field]
            for index in list(rankings[model])[:primary]:
                item_id = int(item_ids[int(index)])
                if item_id in client.test:
                    band_counts[model][hit_bands[item_id]] += 1
    return {"per_user": per_user, "seed_primary_ndcg": seed_primary, "hit_bands": band_counts}


def stratified_means(
    values: np.ndarray, user_ids: list[int], assignment: dict[int, str], labels: list[str]
) -> dict[str, object]:
    """Mean of a per-user metric within each stratum, with support counts."""
    output: dict[str, object] = {}
    for label in labels:
        mask = np.array([assignment[user_id] == label for user_id in user_ids])
        count = int(mask.sum())
        output[label] = {
            "users": count,
            "mean": float(values[mask].mean()) if count else None,
        }
    return output


def _train_als_seed(
    payload: tuple[tuple, int, float, float, int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Worker entry point: rebuild the CSR matrix, then train one seed."""
    (data, indices, indptr, shape), factors, regularization, alpha, iterations, seed = payload
    matrix = sparse.csr_matrix((data, indices, indptr), shape=shape)
    return train_implicit_als(matrix, factors, regularization, alpha, iterations, seed)


def train_als_runs(
    matrix: sparse.csr_matrix, als_config: dict[str, object], workers: int = 1
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Train one ALS model per seed. Seeds are independent, so they can run in
    separate processes; each worker inherits the single-thread BLAS cap, which
    is what makes the parallel speedup real rather than oversubscription."""
    seeds = [int(value) for value in als_config.get("seeds", [])]
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        fail("als.seeds must list at least two distinct seeds")
    settings = (
        int(als_config["factors"]),
        float(als_config["regularization"]),
        float(als_config["alpha"]),
        int(als_config["iterations"]),
    )
    if workers <= 1:
        return [train_implicit_als(matrix, *settings, seed) for seed in seeds]
    payloads = [
        ((matrix.data, matrix.indices, matrix.indptr, matrix.shape), *settings, seed)
        for seed in seeds
    ]
    with ProcessPoolExecutor(max_workers=min(workers, len(seeds))) as pool:
        return list(pool.map(_train_als_seed, payloads))


def seed_variance_control(
    seed_ndcg: np.ndarray, seeds: list[int]
) -> dict[str, object]:
    pairs: list[dict[str, object]] = []
    for left in range(len(seeds)):
        for right in range(left + 1, len(seeds)):
            difference = seed_ndcg[left] - seed_ndcg[right]
            pairs.append({
                "seeds": [seeds[left], seeds[right]],
                "mean_absolute_ndcg_difference": float(np.abs(difference).mean()),
                "mean_ndcg_difference": float(difference.mean()),
            })
    per_seed = [float(seed_ndcg[index].mean()) for index in range(len(seeds))]
    return {
        "scope": "identical full-reference training data, different initialization seeds",
        "per_seed_mean_ndcg_at_10": dict(zip((str(seed) for seed in seeds), per_seed)),
        "mean_ndcg_spread": max(per_seed) - min(per_seed),
        "pairs": pairs,
        "max_mean_absolute_ndcg_difference": max(
            float(pair["mean_absolute_ndcg_difference"]) for pair in pairs
        ),
    }


def bootstrap_interval(
    values: np.ndarray, replicates: int, seed: int, block: int = 200
) -> tuple[float, float]:
    """Paired client-level bootstrap interval, resampled in vectorized blocks."""
    count = len(values)
    if count == 0:
        fail("cannot bootstrap an empty vector")
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, block):
        width = min(block, replicates - start)
        draws = rng.integers(0, count, size=(width, count))
        means[start : start + width] = values[draws].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def policy_metadata(
    selected: dict[int, list[Event]], pilot_counts: Counter[int], tail_ceiling: int,
    clients: dict[int, Client],
) -> dict[str, object]:
    selected_events_flat = [event for events in selected.values() for event in events]
    collection_clients = sum(bool(client.collection) for client in clients.values())
    selected_clients = sum(bool(events) for events in selected.values())
    tail_events = [
        event for event in selected_events_flat
        if pilot_counts.get(event.item_id, 0) <= tail_ceiling
    ]
    return {
        "selected_collection_interactions": len(selected_events_flat),
        "selected_collection_clients": selected_clients,
        "collection_clients": collection_clients,
        "collection_client_coverage": selected_clients / collection_clients if collection_clients else 0.0,
        "selected_tail_interactions": len(tail_events),
        "selected_tail_items": len({event.item_id for event in tail_events}),
        "selected_tail_share": len(tail_events) / len(selected_events_flat) if selected_events_flat else 0.0,
    }


def write_report(output_dir: Path, summary: dict[str, object]) -> None:
    reference = summary["reference"]
    rows = summary["rows"]
    if not isinstance(reference, dict) or not isinstance(rows, list):
        raise AssertionError("invalid summary shape")
    primary = int(reference["primary_cutoff"])
    cutoffs = [int(value) for value in reference["cutoffs"]]
    full_metrics = reference["full_reference_metrics"]
    ceilings = reference["evaluation_ceilings"]
    lines = [
        "# Fixed-cohort local-data-budget replay (protocol v2)",
        "",
        "Status: real MovieLens offline chronological replay; not a federated-system, privacy, or external-generalization result.",
        "",
        "## Design",
        "",
        f"- Fixed deterministic cohort: {reference['cohort_user_count']:,} of {reference['pilot_eligible_user_count']:,} pilot-eligible users; {reference['evaluated_user_count']:,} have future positives.",
        f"- Global positive-rating windows: pilot through {reference['pilot_cutoff_utc']}; collection through {reference['collection_cutoff_utc']}; future test thereafter.",
        f"- Full reference training interactions: {reference['full_training_interactions']:,}, of which {reference['full_collection_interactions']:,} are collection-window events the policies control.",
        f"- Cutoffs {cutoffs} with primary depth {primary}; the display depth is {reference['display_cutoff']}.",
        f"- Models: item-item cosine is the deterministic primary probe over {reference['item_item']['supported_item_count']:,} items with support >= {reference['item_item']['support_threshold']}; implicit ALS is the stochastic secondary probe.",
        "- Full-reference NDCG@%d: " % primary
        + "; ".join(f"{model} {full_metrics[model][str(primary)]['ndcg']:.4f}" for model in full_metrics)
        + ".",
        f"- ALS control: per-user metrics are averaged over seeds {reference['als']['seeds']}; the same-data seed floor is {reference['als']['seed_variance_control']['max_mean_absolute_ndcg_difference']:.4f} mean per-user absolute NDCG difference with a {reference['als']['seed_variance_control']['mean_ndcg_spread']:.4f} mean-NDCG spread.",
        f"- Propensity weighting: {reference['propensity']['model']}, gamma {reference['propensity']['gamma']}, floor {reference['propensity']['minimum_propensity']}, maximum inverse weight {reference['propensity']['max_inverse_weight']:.1f}. The exposure model is an assumption, so weighted values never replace unweighted ones.",
        f"- Metric ceilings: {ceilings['users_with_relevant_above_cutoff']:,} of {reference['evaluated_user_count']:,} users ({ceilings['share_with_relevant_above_cutoff']:.1%}) have more than {primary} future positives; mean recall ceiling {ceilings['mean_recall_ceiling']:.3f}, minimum {ceilings['min_recall_ceiling']:.3f}.",
        "- Policies differ only in collection-window local-record retention; evaluation excludes the same full pre-test seen set for every policy.",
        "",
        "## Policy evidence at the primary cutoff",
        "",
        f"| Policy | Cap | Model | NDCG@{primary} | Weighted NDCG@{primary} | Abs. NDCG error | 95% CI of NDCG delta | Precision@{primary} | Recall@{primary} | Collection rows | Tail share | Tail-vs-equal improvement (95% CI) |",
        "| --- | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['policy']} | {row['cap']} | {row['model']} | "
            f"{row['mean_ndcg']:.4f} | {row['mean_weighted_ndcg']:.4f} | "
            f"{row['mean_absolute_ndcg_error']:.4f} | "
            f"[{row['ndcg_delta_ci_low']:.4f}, {row['ndcg_delta_ci_high']:.4f}] | "
            f"{row['mean_precision']:.4f} | {row['mean_recall']:.4f} | "
            f"{row['selected_collection_interactions']:,} | {row['selected_tail_share']:.3f} | "
            f"{row['tail_vs_equal_absolute_error_improvement']:.4f} "
            f"[{row['tail_vs_equal_absolute_error_improvement_ci_low']:.4f}, "
            f"{row['tail_vs_equal_absolute_error_improvement_ci_high']:.4f}] |")
    lines += [
        "",
        "## Cutoff sensitivity (equal chronological cap)",
        "",
        "| Cap | Model | " + " | ".join(f"NDCG@{cutoff}" for cutoff in cutoffs) + " |",
        "| ---: | --- | " + " | ".join("---:" for _ in cutoffs) + " |",
    ]
    for row in rows:
        if row["policy"] != "equal_chronological_cap":
            continue
        values = " | ".join(
            f"{row['metrics_by_cutoff'][str(cutoff)]['ndcg']:.4f}" for cutoff in cutoffs
        )
        lines.append(f"| {row['cap']} | {row['model']} | {values} |")
    activity_labels = list(next(iter(rows))["strata"]["user_activity"].keys())
    band_labels = list(next(iter(rows))["strata"]["hit_popularity_bands"].keys())
    lines += [
        "",
        "## Stratified evidence at the primary cutoff",
        "",
        "| Policy | Cap | Model | "
        + " | ".join(f"NDCG {label}" for label in activity_labels)
        + " | " + " | ".join(f"hits {label}" for label in band_labels) + " |",
        "| --- | ---: | --- | " + " | ".join("---:" for _ in activity_labels + band_labels) + " |",
    ]
    for row in rows:
        activity_cells = " | ".join(
            "n/a" if row["strata"]["user_activity"][label]["mean"] is None
            else f"{row['strata']['user_activity'][label]['mean']:.4f}"
            for label in activity_labels
        )
        band_cells = " | ".join(
            f"{row['strata']['hit_popularity_bands'][label]:,}" for label in band_labels
        )
        lines.append(
            f"| {row['policy']} | {row['cap']} | {row['model']} | {activity_cells} | {band_cells} |"
        )
    tail_rows = [row for row in rows if row["policy"] == "tail_reserve_cap"]
    tail_improvements = [
        row for row in tail_rows
        if float(row["tail_vs_equal_absolute_error_improvement_ci_low"]) > 0.0
    ]
    tail_conclusion = (
        "- At least one tested tail-reserve cell has a strictly positive paired "
        "95% interval; inspect the complete grid before promoting it."
        if tail_improvements
        else "- No tested tail-reserve cell has a strictly positive paired 95% "
        "interval for lower absolute NDCG error than the equal chronological cap."
    )
    noise_floor = float(
        reference["als"]["seed_variance_control"]["max_mean_absolute_ndcg_difference"]
    )
    als_above_floor = [
        row for row in rows
        if row["model"] == "implicit_als"
        and float(row["mean_absolute_ndcg_error"]) > noise_floor
    ]
    als_conclusion = (
        f"- {len(als_above_floor)} of "
        f"{sum(1 for row in rows if row['model'] == 'implicit_als')} ALS cells exceed the "
        "same-data seed-variance floor, so their per-user error is not explained by "
        "optimizer initialization alone."
        if als_above_floor
        else "- No ALS cell exceeds the same-data seed-variance floor, so per-user ALS "
        "differences here are not separable from optimizer initialization noise."
    )
    item_item_rows = [row for row in rows if row["model"] == "item_item_cosine"]
    item_item_separating = [
        row for row in item_item_rows
        if float(row["ndcg_delta_ci_high"]) < 0.0 or float(row["ndcg_delta_ci_low"]) > 0.0
    ]
    item_item_conclusion = (
        f"- Item-item cosine has zero model-noise floor and separates the budget in "
        f"{len(item_item_separating)} of {len(item_item_rows)} cells, where the paired 95% "
        "NDCG interval against full history excludes zero."
        if item_item_separating
        else "- Item-item cosine has zero model-noise floor, yet no cell's paired 95% NDCG "
        "interval against full history excludes zero: the tested caps do not measurably "
        "change this personalized model."
    )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "- A positive tail-vs-equal improvement means the tail-reserve policy had lower mean per-user absolute NDCG error than the equal chronological cap at the same cap.",
        tail_conclusion,
        item_item_conclusion,
        als_conclusion,
        "- Mean NDCG stability and per-user stability are different claims; report both.",
        "- The configured 0.005 mean-absolute-NDCG tolerance is exploratory, not a product-risk threshold or a validated stop-controller rule.",
        "- The replay validates policy-specific sample-to-reference behavior on this cohort and time partition only. It does not validate client availability, dropout, communication, secure aggregation, local compute, consent, or federated convergence.",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def load_or_build_cohort(
    chunks: list[Path], manifest_path: Path, windows: dict[str, object],
    cohort_config: dict[str, object], cache_dir: Path,
) -> dict[str, object]:
    """Build the fixed cohort, or reuse a cached build of the same contract.

    The three passes over 33.8 million ratings dominate a warm rerun, and they
    depend only on the dataset manifest plus the window and cohort settings.
    """
    key_payload = json.dumps(
        {
            "manifest_sha256": sha256_file(manifest_path),
            "chunks": [str(path.relative_to(ROOT)) for path in chunks],
            "windows": windows,
            "cohort": cohort_config,
            "version": 1,
        },
        sort_keys=True,
    ).encode()
    key = hashlib.sha256(key_payload).hexdigest()[:24]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"cohort_{key}.pkl"
    if cache_path.is_file():
        print(f"reusing cached cohort build {key}", flush=True)
        with cache_path.open("rb") as handle:
            return pickle.load(handle)

    print("computing global chronological cutoffs", flush=True)
    pilot_cutoff, collection_cutoff, global_counts, catalogue = global_cutoffs(
        chunks, float(windows["pilot_fraction"]), float(windows["collection_fraction"])
    )
    print("selecting fixed recent pilot-eligible cohort", flush=True)
    cohort, pilot_eligible_count = select_cohort(
        chunks,
        pilot_cutoff,
        int(cohort_config["minimum_pilot_positive_interactions"]),
        int(cohort_config["recent_pilot_activity_days"]),
        int(cohort_config["panel_size"]),
        int(cohort_config["seed"]),
    )
    print("loading fixed cohort histories", flush=True)
    clients = load_clients(
        chunks, cohort, pilot_cutoff, collection_cutoff, int(cohort_config["seed"])
    )
    state = {
        "cache_key": key,
        "pilot_cutoff": pilot_cutoff,
        "collection_cutoff": collection_cutoff,
        "global_counts": global_counts,
        "catalogue": catalogue,
        "cohort": cohort,
        "pilot_eligible_count": pilot_eligible_count,
        "clients": clients,
    }
    with cache_path.open("wb") as handle:
        pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/experiments/fixed_cohort_budget_v2.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/explorations/fixed_cohort_budget_v2"),
    )
    parser.add_argument(
        "--panel-size", type=int, default=None,
        help="override the configured cohort size; smoke tests only",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("artifacts/fixed_cohort_budget_cache"),
        help="cohort-construction cache; delete it to force a full rebuild",
    )
    parser.add_argument(
        "--als-workers", type=int, default=min(3, os.cpu_count() or 1),
        help="processes used for the independent ALS seed trainings (1 disables)",
    )
    args = parser.parse_args()

    config_path = repo_path(args.config)
    output_dir = repo_path(args.output)
    config = load_json(config_path)
    dataset = nested(config, "dataset")
    windows = nested(config, "windows")
    cohort_config = nested(config, "cohort")
    tail_config = nested(config, "tail")
    evaluation = nested(config, "evaluation")
    als_config = nested(config, "als")
    item_item_config = nested(config, "item_item")
    propensity_config = nested(config, "propensity")
    strata_config = nested(config, "strata")
    if args.panel_size is not None:
        cohort_config = {**cohort_config, "panel_size": int(args.panel_size)}
    manifest_path = repo_path(str(dataset["manifest"]))
    ratings_dir = repo_path(str(dataset["ratings_dir"]))
    if not manifest_path.is_file():
        fail(f"missing manifest: {manifest_path}")
    chunks = rating_chunks(ratings_dir)
    caps = tuple(int(value) for value in config.get("caps", []))
    if not caps or any(value <= 0 for value in caps) or tuple(sorted(set(caps))) != caps:
        fail("caps must be a strictly increasing positive integer list")
    models = tuple(str(value) for value in evaluation.get("models", []))
    if models != MODELS:
        fail(f"this replay requires models {MODELS}, got {models}")
    cutoffs = tuple(int(value) for value in evaluation.get("cutoffs", []))
    primary_cutoff = int(evaluation["primary_cutoff"])
    if not cutoffs or tuple(sorted(set(cutoffs))) != cutoffs:
        fail("cutoffs must be a strictly increasing positive integer list")
    if primary_cutoff != max(cutoffs):
        fail("the primary cutoff must be the deepest configured cutoff")

    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    cohort_state = load_or_build_cohort(
        chunks, manifest_path, windows, cohort_config, repo_path(args.cache_dir)
    )
    pilot_cutoff = cohort_state["pilot_cutoff"]
    collection_cutoff = cohort_state["collection_cutoff"]
    global_counts = cohort_state["global_counts"]
    catalogue = cohort_state["catalogue"]
    cohort = cohort_state["cohort"]
    pilot_eligible_count = cohort_state["pilot_eligible_count"]
    clients = cohort_state["clients"]
    item_ids = np.asarray(sorted(catalogue), dtype=np.int64)
    item_index = {int(item_id): index for index, item_id in enumerate(item_ids)}
    user_row = {user_id: index for index, user_id in enumerate(cohort)}
    pilot_counts = Counter(
        event.item_id for client in clients.values() for event in client.pilot
    )
    tail_ceiling = tail_count_ceiling(
        pilot_counts, float(tail_config["pilot_item_count_quantile"])
    )
    all_collection = {user_id: client.collection for user_id, client in clients.items()}
    print("building full permitted-history reference", flush=True)
    full_counts, full_sums, full_interactions, full_prior = training_statistics(
        clients, all_collection, item_index
    )
    full_orders = rank_orders(
        full_counts, full_sums, item_ids, full_prior, float(evaluation["smoothing_count"])
    )
    als_seeds = [int(value) for value in als_config.get("seeds", [])]
    evaluated_users = [user_id for user_id in cohort if clients[user_id].test]
    if not evaluated_users:
        fail("no cohort users have future positive test items after seen-item exclusion")
    evaluated_rows = [user_row[user_id] for user_id in evaluated_users]
    seen_masks = []
    for user_id in evaluated_users:
        mask = np.zeros(len(item_ids), dtype=bool)
        seen_indices = [item_index[item] for item in clients[user_id].seen if item in item_index]
        if seen_indices:
            mask[np.asarray(seen_indices, dtype=np.int64)] = True
        seen_masks.append(mask)
    support_threshold = int(item_item_config["support_threshold"])
    support_indices = np.flatnonzero(full_counts >= support_threshold).astype(np.int64)
    block_size = int(item_item_config["candidate_block_size"])
    weights = propensity_weights(full_counts, item_ids, propensity_config)
    activity_labels = list(nested(strata_config, "user_activity")["labels"])
    band_labels = list(nested(strata_config, "item_popularity")["labels"])
    size_labels = list(nested(strata_config, "relevant_set_size")["labels"])
    activity = activity_strata(clients, evaluated_users, activity_labels)
    hit_bands = popularity_bands(full_counts, item_ids, band_labels)
    relevant_size_strata = {
        user_id: size_labels[0] if len(clients[user_id].test) <= primary_cutoff else size_labels[1]
        for user_id in evaluated_users
    }
    print("building full permitted-history item-item cosine reference", flush=True)
    full_matrix = training_matrix(clients, all_collection, user_row, item_index)
    full_item_item = item_item_scores(
        full_matrix, evaluated_rows, support_indices, seen_masks, item_ids,
        primary_cutoff, block_size,
    )
    print(f"training full-reference implicit ALS over {len(als_seeds)} seeds", flush=True)
    full_als = train_als_runs(full_matrix, als_config, args.als_workers)
    full_evaluation = evaluate_condition(
        full_orders, full_item_item, full_als, evaluated_users, user_row, item_ids,
        seen_masks, clients, weights, cutoffs, hit_bands, band_labels,
    )
    full_metrics = full_evaluation["per_user"]
    als_control = seed_variance_control(full_evaluation["seed_primary_ndcg"], als_seeds)

    reference = {
        "scope": "all permitted positive pilot and collection-window history from fixed cohort",
        "protocol": "v2",
        "cohort_user_count": len(cohort),
        "pilot_eligible_user_count": pilot_eligible_count,
        "evaluated_user_count": len(evaluated_users),
        "cohort_users_without_future_positive_after_seen_exclusion": len(cohort) - len(evaluated_users),
        "pilot_cutoff_timestamp": pilot_cutoff,
        "collection_cutoff_timestamp": collection_cutoff,
        "pilot_cutoff_utc": datetime.fromtimestamp(pilot_cutoff, timezone.utc).isoformat(),
        "collection_cutoff_utc": datetime.fromtimestamp(collection_cutoff, timezone.utc).isoformat(),
        "global_counts": global_counts,
        "full_training_interactions": full_interactions,
        "full_collection_interactions": sum(len(client.collection) for client in clients.values()),
        "pilot_training_interactions": sum(len(client.pilot) for client in clients.values()),
        "tail_count_ceiling": tail_ceiling,
        "catalogue_item_count": len(item_ids),
        "cutoffs": list(cutoffs),
        "primary_cutoff": primary_cutoff,
        "display_cutoff": int(evaluation["display_cutoff"]),
        "propensity": {
            **{key: propensity_config[key] for key in ("estimator", "model", "gamma", "minimum_propensity", "reporting_rule")},
            "max_inverse_weight": float(max(weights.values())),
        },
        "strata": {
            "user_activity": {
                label: sum(1 for value in activity.values() if value == label)
                for label in activity_labels
            },
            "relevant_set_size": {
                label: sum(1 for value in relevant_size_strata.values() if value == label)
                for label in size_labels
            },
            "item_popularity_bands": {
                label: sum(1 for value in hit_bands.values() if value == label)
                for label in band_labels
            },
        },
        "item_item": {
            "support_threshold": support_threshold,
            "supported_item_count": int(len(support_indices)),
            "top_k": int(item_item_config["top_k"]),
            "candidate_block_size": block_size,
            "support_scope": str(item_item_config["support_scope"]),
            "determinism": "no random state; identical inputs reproduce identical scores",
        },
        "als": {
            "factors": int(als_config["factors"]),
            "regularization": float(als_config["regularization"]),
            "alpha": float(als_config["alpha"]),
            "iterations": int(als_config["iterations"]),
            "seeds": als_seeds,
            "aggregation": str(als_config["aggregation"]),
            "tuning": "fixed a priori; no hyperparameter search against the future-test window",
            "seed_variance_control": als_control,
        },
        "full_reference_metrics": {
            model: {
                str(cutoff): {
                    field: float(full_metrics[model][cutoff][field].mean())
                    for field in ("ndcg", "precision", "recall", "weighted_ndcg", "weighted_recall")
                }
                for cutoff in cutoffs
            }
            for model in MODELS
        },
        "full_reference_strata": {
            model: {
                "user_activity": stratified_means(
                    full_metrics[model][primary_cutoff]["ndcg"], evaluated_users, activity, activity_labels
                ),
                "relevant_set_size": stratified_means(
                    full_metrics[model][primary_cutoff]["ndcg"], evaluated_users, relevant_size_strata, size_labels
                ),
                "hit_popularity_bands": full_evaluation["hit_bands"][model],
            }
            for model in MODELS
        },
        "evaluation_ceilings": evaluation_ceilings(clients, evaluated_users, primary_cutoff),
    }
    (output_dir / "reference_artifact.json").write_text(
        json.dumps(reference, indent=2) + "\n", encoding="utf-8"
    )

    rows: list[dict[str, object]] = []
    ndcg_by_policy: dict[tuple[str, int, str], np.ndarray] = {}
    metadata_by_policy: dict[tuple[str, int], dict[str, object]] = {}
    for cap in caps:
        for policy in POLICIES:
            print(f"evaluating {policy} cap={cap}", flush=True)
            selected = selected_events(
                policy, clients, cap, pilot_counts, tail_ceiling,
                float(tail_config["reserve_fraction"]),
            )
            metadata = policy_metadata(selected, pilot_counts, tail_ceiling, clients)
            metadata_by_policy[(policy, cap)] = metadata
            counts, sums, interactions, prior = training_statistics(clients, selected, item_index)
            orders = rank_orders(
                counts, sums, item_ids, prior, float(evaluation["smoothing_count"])
            )
            policy_matrix = training_matrix(clients, selected, user_row, item_index)
            policy_item_item = item_item_scores(
                policy_matrix, evaluated_rows, support_indices, seen_masks, item_ids,
                primary_cutoff, block_size,
            )
            policy_als = train_als_runs(policy_matrix, als_config, args.als_workers)
            condition = evaluate_condition(
                orders, policy_item_item, policy_als, evaluated_users, user_row, item_ids,
                seen_masks, clients, weights, cutoffs, hit_bands, band_labels,
            )
            metrics = condition["per_user"]
            for model_index, model in enumerate(MODELS):
                primary = metrics[model][primary_cutoff]["ndcg"]
                reference_primary = full_metrics[model][primary_cutoff]["ndcg"]
                delta = primary - reference_primary
                ci_low, ci_high = bootstrap_interval(
                    delta,
                    int(evaluation["bootstrap_replicates"]),
                    int(evaluation["bootstrap_seed"]) + cap * 100 + model_index,
                )
                ndcg_by_policy[(policy, cap, model)] = primary
                rows.append({
                    "policy": policy,
                    "cap": cap,
                    "model": model,
                    "training_interactions": interactions,
                    **metadata,
                    "primary_cutoff": primary_cutoff,
                    "ndcg_delta_by_cutoff": {
                        str(cutoff): {
                            "delta": float(
                                (metrics[model][cutoff]["ndcg"]
                                 - full_metrics[model][cutoff]["ndcg"]).mean()
                            ),
                            "ci_low": bootstrap_interval(
                                metrics[model][cutoff]["ndcg"] - full_metrics[model][cutoff]["ndcg"],
                                int(evaluation["bootstrap_replicates"]),
                                int(evaluation["bootstrap_seed"]) + cap * 100 + model_index * 7 + cutoff,
                            )[0],
                            "ci_high": bootstrap_interval(
                                metrics[model][cutoff]["ndcg"] - full_metrics[model][cutoff]["ndcg"],
                                int(evaluation["bootstrap_replicates"]),
                                int(evaluation["bootstrap_seed"]) + cap * 100 + model_index * 7 + cutoff,
                            )[1],
                        }
                        for cutoff in cutoffs
                    },
                    "metrics_by_cutoff": {
                        str(cutoff): {
                            field: float(metrics[model][cutoff][field].mean())
                            for field in ("ndcg", "precision", "recall", "weighted_ndcg", "weighted_recall")
                        }
                        for cutoff in cutoffs
                    },
                    "mean_ndcg": float(primary.mean()),
                    "mean_weighted_ndcg": float(metrics[model][primary_cutoff]["weighted_ndcg"].mean()),
                    "mean_precision": float(metrics[model][primary_cutoff]["precision"].mean()),
                    "mean_recall": float(metrics[model][primary_cutoff]["recall"].mean()),
                    "mean_ndcg_delta_vs_full": float(delta.mean()),
                    "mean_absolute_ndcg_error": float(np.abs(delta).mean()),
                    "ndcg_delta_ci_low": ci_low,
                    "ndcg_delta_ci_high": ci_high,
                    "mean_weighted_ndcg_delta_vs_full": float(
                        (metrics[model][primary_cutoff]["weighted_ndcg"]
                         - full_metrics[model][primary_cutoff]["weighted_ndcg"]).mean()
                    ),
                    "strata": {
                        "user_activity": stratified_means(
                            primary, evaluated_users, activity, activity_labels
                        ),
                        "relevant_set_size": stratified_means(
                            primary, evaluated_users, relevant_size_strata, size_labels
                        ),
                        "hit_popularity_bands": condition["hit_bands"][model],
                    },
                    "exploratory_tolerance_pass": bool(
                        float(np.abs(delta).mean()) <= float(evaluation["exploratory_mean_absolute_ndcg_tolerance"])
                    ),
                })

    for row in rows:
        equal = ndcg_by_policy[("equal_chronological_cap", int(row["cap"]), str(row["model"]))]
        tail = ndcg_by_policy[("tail_reserve_cap", int(row["cap"]), str(row["model"]))]
        reference_ndcg = full_metrics[str(row["model"])][primary_cutoff]["ndcg"]
        improvement = np.abs(equal - reference_ndcg) - np.abs(tail - reference_ndcg)
        ci_low, ci_high = bootstrap_interval(
            improvement,
            int(evaluation["bootstrap_replicates"]),
            int(evaluation["bootstrap_seed"]) + int(row["cap"]) * 1000 + MODELS.index(str(row["model"])),
        )
        row["tail_vs_equal_absolute_error_improvement"] = float(improvement.mean())
        row["tail_vs_equal_absolute_error_improvement_ci_low"] = ci_low
        row["tail_vs_equal_absolute_error_improvement_ci_high"] = ci_high

    summary = {
        "run_id": "fixed_cohort_budget_v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": {
            "path": str(Path(__file__).relative_to(ROOT)),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "inputs": {
            "config_path": str(config_path.relative_to(ROOT)),
            "config_sha256": sha256_file(config_path),
            "manifest_path": str(manifest_path.relative_to(ROOT)),
            "manifest_sha256": sha256_file(manifest_path),
            "ratings_chunks": [str(path.relative_to(ROOT)) for path in chunks],
        },
        "design": config,
        "reference": reference,
        "rows": rows,
        "runtime_seconds": time.perf_counter() - started,
        "interpretation": {
            "supported_claim": "policy-specific fixed-cohort sample-to-reference fidelity on the pinned MovieLens replay",
            "not_supported": [
                "federated convergence", "device availability", "dropout", "communication cost",
                "secure aggregation", "privacy proof", "consent", "external-domain transfer",
                "validated stop-controller safety",
            ],
        },
    }
    (output_dir / "candidate_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    write_report(output_dir, summary)
    print(f"created {output_dir / 'candidate_summary.json'}", flush=True)
    print(f"created {output_dir / 'report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
