#!/usr/bin/env python3
"""Select a deterministic pilot user set and support filtered item catalog."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Iterable, NoReturn


REQUIRED_INTERACTION_COLUMNS = {"user_id", "item_id"}
REQUIRED_CLUSTER_COLUMNS = {"user_id", "cluster_id"}


def fail(message: str) -> NoReturn:
    raise SystemExit(f"error: {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interactions", type=Path, required=True, help="training only interaction CSV")
    parser.add_argument("--cluster-map", type=Path, required=True, help="training only user to oracle cluster CSV")
    parser.add_argument("--output", type=Path, required=True, help="pilot selection JSON")
    parser.add_argument("--users", type=int, default=10000)
    parser.add_argument("--clusters", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--min-user-interactions", type=int, default=20)
    parser.add_argument("--min-item-support-per-cluster", type=int, default=20)
    parser.add_argument("--min-eligible-items-per-cluster", "--min-eligible-items", dest="min_eligible_items", type=int, default=1000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def interaction_rows(path: Path) -> Iterable[tuple[int, int]]:
    try:
        handle = path.open("r", encoding="utf-8", newline="")
    except OSError as exc:
        fail(f"cannot open interactions file {path}: {exc}")
    with handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        if not REQUIRED_INTERACTION_COLUMNS.issubset(columns):
            fail(f"interactions file needs columns {sorted(REQUIRED_INTERACTION_COLUMNS)}")
        for line_number, row in enumerate(reader, start=2):
            try:
                yield int(row["user_id"]), int(row["item_id"])
            except (TypeError, ValueError) as exc:
                fail(f"invalid interaction at line {line_number}: {exc}")


def cluster_map(path: Path, cluster_count: int) -> dict[int, int]:
    try:
        handle = path.open("r", encoding="utf-8", newline="")
    except OSError as exc:
        fail(f"cannot open cluster map {path}: {exc}")
    result: dict[int, int] = {}
    with handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        if not REQUIRED_CLUSTER_COLUMNS.issubset(columns):
            fail(f"cluster map needs columns {sorted(REQUIRED_CLUSTER_COLUMNS)}")
        for line_number, row in enumerate(reader, start=2):
            try:
                user_id = int(row["user_id"])
                cluster_id = int(row["cluster_id"])
            except (TypeError, ValueError) as exc:
                fail(f"invalid cluster map at line {line_number}: {exc}")
            if cluster_id < 0 or cluster_id >= cluster_count:
                fail(f"cluster {cluster_id} at line {line_number} is outside 0 through {cluster_count - 1}")
            if user_id in result and result[user_id] != cluster_id:
                fail(f"user {user_id} has conflicting cluster assignments")
            result[user_id] = cluster_id
    return result


def percentile(values: list[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def main() -> int:
    args = parse_args()
    if args.users <= 0 or args.clusters <= 0:
        fail("users and clusters must be positive")
    if args.users % args.clusters != 0:
        fail("users must divide evenly across clusters for stratified sampling")
    if args.min_user_interactions <= 0 or args.min_item_support_per_cluster <= 0:
        fail("support thresholds must be positive")
    if args.output.exists() and not args.force:
        fail(f"output exists: {args.output}; use --force to replace it")

    assignments = cluster_map(args.cluster_map, args.clusters)
    user_counts: Counter[int] = Counter()
    for user_id, _ in interaction_rows(args.interactions):
        user_counts[user_id] += 1

    target_per_cluster = args.users // args.clusters
    eligible_users_by_cluster: dict[int, list[int]] = {cluster_id: [] for cluster_id in range(args.clusters)}
    for user_id, count in user_counts.items():
        if count >= args.min_user_interactions and user_id in assignments:
            eligible_users_by_cluster[assignments[user_id]].append(user_id)
    for cluster_id in eligible_users_by_cluster:
        eligible_users_by_cluster[cluster_id].sort()
        if len(eligible_users_by_cluster[cluster_id]) < target_per_cluster:
            fail(
                f"cluster {cluster_id} has {len(eligible_users_by_cluster[cluster_id])} eligible users; "
                f"need {target_per_cluster}"
            )

    rng = random.Random(args.seed)
    selected_users_by_cluster = {
        cluster_id: sorted(rng.sample(eligible_users_by_cluster[cluster_id], target_per_cluster))
        for cluster_id in range(args.clusters)
    }
    selected_users = sorted(
        user_id
        for users in selected_users_by_cluster.values()
        for user_id in users
    )
    selected_set = set(selected_users)

    item_cluster_counts: dict[int, list[int]] = defaultdict(lambda: [0] * args.clusters)
    cluster_users: list[set[int]] = [set() for _ in range(args.clusters)]
    cluster_ratings = [0] * args.clusters
    observed_items: set[int] = set()
    for user_id, item_id in interaction_rows(args.interactions):
        if user_id not in selected_set:
            continue
        cluster_id = assignments[user_id]
        item_cluster_counts[item_id][cluster_id] += 1
        cluster_users[cluster_id].add(user_id)
        cluster_ratings[cluster_id] += 1
        observed_items.add(item_id)

    eligible_items_by_cluster: dict[int, list[int]] = {
        cluster_id: sorted(
            item_id
            for item_id in observed_items
            if item_cluster_counts[item_id][cluster_id] >= args.min_item_support_per_cluster
        )
        for cluster_id in range(args.clusters)
    }
    intersection = set.intersection(
        *(set(items) for items in eligible_items_by_cluster.values())
    )
    below_threshold_by_cluster = {
        str(cluster_id): sum(
            item_cluster_counts[item_id][cluster_id] < args.min_item_support_per_cluster
            for item_id in observed_items
        )
        for cluster_id in range(args.clusters)
    }
    zero_cluster_support = sum(
        any(item_cluster_counts[item_id][cluster_id] == 0 for cluster_id in range(args.clusters))
        for item_id in observed_items
    )
    per_cluster = []
    for cluster_id in range(args.clusters):
        eligible_items = eligible_items_by_cluster[cluster_id]
        supports = [item_cluster_counts[item_id][cluster_id] for item_id in eligible_items]
        per_cluster.append(
            {
                "cluster_id": cluster_id,
                "rating_count": cluster_ratings[cluster_id],
                "user_count": len(cluster_users[cluster_id]),
                "item_count": len(supports),
                "support_minimum": min(supports, default=0),
                "support_median": median(supports) if supports else 0,
                "support_p10": percentile(supports, 0.10),
                "support_p90": percentile(supports, 0.90),
            }
        )

    support_passed = all(
        len(items) >= args.min_eligible_items
        for items in eligible_items_by_cluster.values()
    )
    result = {
        "status": "passed" if support_passed else "insufficient_support",
        "selection": {
            "seed": args.seed,
            "sampling": "stratified_by_oracle_cluster",
            "target_user_count": args.users,
            "target_users_per_cluster": target_per_cluster,
            "selected_user_count": len(selected_users),
            "minimum_user_interactions": args.min_user_interactions,
            "minimum_item_support_per_cluster": args.min_item_support_per_cluster,
            "minimum_eligible_items_per_cluster": args.min_eligible_items,
            "cluster_count": args.clusters,
            "source_is_training_only": True,
            "cluster_selection_counts": {
                str(cluster_id): len(users)
                for cluster_id, users in selected_users_by_cluster.items()
            },
        },
        "selected_user_ids": selected_users,
        "selected_user_ids_by_cluster": {
            str(cluster_id): users
            for cluster_id, users in selected_users_by_cluster.items()
        },
        "eligible_item_ids_by_cluster": {
            str(cluster_id): items
            for cluster_id, items in eligible_items_by_cluster.items()
        },
        "eligible_item_intersection_ids": sorted(intersection),
        "support_report": {
            "support_check_passed": support_passed,
            "eligible_item_count_by_cluster": {
                str(cluster_id): len(items)
                for cluster_id, items in eligible_items_by_cluster.items()
            },
            "eligible_item_intersection_count": len(intersection),
            "observed_item_count": len(observed_items),
            "items_below_support_threshold_by_cluster": below_threshold_by_cluster,
            "items_with_zero_cluster_support": zero_cluster_support,
            "per_cluster": per_cluster,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"created {args.output}")
    print(f"selected users: {len(selected_users):,}")
    print(
        "eligible items by cluster: "
        + ", ".join(
            f"{cluster_id}={len(items):,}"
            for cluster_id, items in eligible_items_by_cluster.items()
        )
    )
    print(f"eligible item intersection: {len(intersection):,}")
    if not support_passed:
        print(
            "support requirement failed; increase the user population or reduce the cluster count",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
