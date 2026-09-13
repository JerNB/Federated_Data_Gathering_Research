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
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from run_sample_generalization import (
    RATING_THRESHOLD,
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
MODELS = ("popularity", "rating_weighted_popularity")


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


def evaluate_orders(
    orders: dict[str, np.ndarray], item_ids: np.ndarray, clients: dict[int, Client], cutoff: int
) -> tuple[list[int], dict[str, np.ndarray], dict[str, np.ndarray]]:
    user_ids = [user_id for user_id, client in clients.items() if client.test]
    if not user_ids:
        fail("no cohort users have future positive test items after seen-item exclusion")
    recall = {model: np.empty(len(user_ids), dtype=np.float64) for model in MODELS}
    ndcg = {model: np.empty(len(user_ids), dtype=np.float64) for model in MODELS}
    for row, user_id in enumerate(user_ids):
        client = clients[user_id]
        for model in MODELS:
            recall_value, ndcg_value = metric_at_k(
                orders[model], item_ids, client.seen, client.test, cutoff
            )
            recall[model][row] = recall_value
            ndcg[model][row] = ndcg_value
    return user_ids, recall, ndcg


def bootstrap_interval(values: np.ndarray, replicates: int, seed: int) -> tuple[float, float]:
    if len(values) == 0:
        fail("cannot bootstrap an empty vector")
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        means[index] = float(values[rng.integers(0, len(values), len(values))].mean())
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
    lines = [
        "# Fixed-cohort local-data-budget replay",
        "",
        "Status: real MovieLens offline chronological replay; not a federated-system, privacy, or external-generalization result.",
        "",
        "## Design",
        "",
        f"- Fixed deterministic cohort: {reference['cohort_user_count']:,} of {reference['pilot_eligible_user_count']:,} pilot-eligible users.",
        f"- Global positive-rating windows: pilot through {reference['pilot_cutoff_utc']}; collection through {reference['collection_cutoff_utc']}; future test thereafter.",
        f"- Full reference training interactions: {reference['full_training_interactions']:,}; evaluated users with future positives: {reference['evaluated_user_count']:,}.",
        f"- Pilot item-tail ceiling: <= {reference['tail_count_ceiling']} positive pilot interactions.",
        "- Policies differ only in collection-window local-record retention; evaluation excludes the same full pre-test seen set for every policy.",
        "",
        "## Policy evidence",
        "",
        "| Policy | Cap | Model | Mean NDCG@10 | Mean abs. NDCG error | 95% CI of NDCG delta vs full | Mean Recall@10 | Collection rows | Tail share | Tail-vs-equal abs.-error improvement (95% CI) |",
        "| --- | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['policy']} | {row['cap']} | {row['model']} | "
            f"{row['mean_ndcg_at_10']:.4f} | {row['mean_absolute_ndcg_error']:.4f} | "
            f"[{row['ndcg_delta_ci_low']:.4f}, {row['ndcg_delta_ci_high']:.4f}] | "
            f"{row['mean_recall_at_10']:.4f} | {row['selected_collection_interactions']:,} | "
            f"{row['selected_tail_share']:.3f} | "
            f"{row['tail_vs_equal_absolute_error_improvement']:.4f} "
            f"[{row['tail_vs_equal_absolute_error_improvement_ci_low']:.4f}, "
            f"{row['tail_vs_equal_absolute_error_improvement_ci_high']:.4f}] |")
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
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "- A positive tail-vs-equal improvement means the tail-reserve policy had lower mean per-user absolute NDCG error than the equal chronological cap at the same cap.",
        tail_conclusion,
        "- The configured 0.005 mean-absolute-NDCG tolerance is exploratory, not a product-risk threshold or a validated stop-controller rule.",
        "- The replay validates policy-specific sample-to-reference behavior on this cohort and time partition only. It does not validate client availability, dropout, communication, secure aggregation, local compute, consent, or federated convergence.",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/experiments/fixed_cohort_budget_v1.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/explorations/fixed_cohort_budget_v1"),
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

    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
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
    item_ids = np.asarray(sorted(catalogue), dtype=np.int64)
    item_index = {int(item_id): index for index, item_id in enumerate(item_ids)}
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
    evaluated_users, full_recall, full_ndcg = evaluate_orders(
        full_orders, item_ids, clients, int(evaluation["cutoff"])
    )
    if len(evaluated_users) != len(set(evaluated_users)):
        fail("evaluation users must be unique")

    reference = {
        "scope": "all permitted positive pilot and collection-window history from fixed cohort",
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
        "full_mean_ndcg_at_10": {model: float(full_ndcg[model].mean()) for model in MODELS},
        "full_mean_recall_at_10": {model: float(full_recall[model].mean()) for model in MODELS},
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
            policy_users, recall, ndcg = evaluate_orders(
                orders, item_ids, clients, int(evaluation["cutoff"])
            )
            if policy_users != evaluated_users:
                fail("policy evaluation changed the fixed test cohort")
            for model_index, model in enumerate(MODELS):
                delta = ndcg[model] - full_ndcg[model]
                ci_low, ci_high = bootstrap_interval(
                    delta,
                    int(evaluation["bootstrap_replicates"]),
                    int(evaluation["bootstrap_seed"]) + cap * 100 + model_index,
                )
                ndcg_by_policy[(policy, cap, model)] = ndcg[model]
                rows.append({
                    "policy": policy,
                    "cap": cap,
                    "model": model,
                    "training_interactions": interactions,
                    **metadata,
                    "mean_ndcg_at_10": float(ndcg[model].mean()),
                    "mean_recall_at_10": float(recall[model].mean()),
                    "mean_ndcg_delta_vs_full": float(delta.mean()),
                    "mean_absolute_ndcg_error": float(np.abs(delta).mean()),
                    "ndcg_delta_ci_low": ci_low,
                    "ndcg_delta_ci_high": ci_high,
                    "mean_recall_delta_vs_full": float((recall[model] - full_recall[model]).mean()),
                    "exploratory_tolerance_pass": bool(
                        float(np.abs(delta).mean()) <= float(evaluation["exploratory_mean_absolute_ndcg_tolerance"])
                    ),
                })

    for row in rows:
        equal = ndcg_by_policy[("equal_chronological_cap", int(row["cap"]), str(row["model"]))]
        tail = ndcg_by_policy[("tail_reserve_cap", int(row["cap"]), str(row["model"]))]
        reference_ndcg = full_ndcg[str(row["model"])]
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
        "run_id": "fixed_cohort_budget_v1",
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
