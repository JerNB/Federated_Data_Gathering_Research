#!/usr/bin/env python3
"""Validate fixed-cohort local-data-budget replay artifacts (protocol v2)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
POLICIES = {"equal_chronological_cap", "tail_reserve_cap"}
MODELS = {
    "popularity",
    "rating_weighted_popularity",
    "item_item_cosine",
    "implicit_als",
}
METRIC_FIELDS = ("ndcg", "precision", "recall", "weighted_ndcg", "weighted_recall")


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


def repo_path(value: Path) -> Path:
    return value if value.is_absolute() else ROOT / value


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot load {path}: {exc}")
    if not isinstance(value, dict):
        fail(f"{path} is not a JSON object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_probability(value: Any, label: str) -> None:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        fail(f"{label} is not finite")
    if not 0.0 <= float(value) <= 1.0:
        fail(f"{label} outside [0, 1]: {value}")


def finite_number(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        fail(f"{label} is not finite")
    return float(value)


def validate_inputs(config_path: Path, summary: dict[str, Any], config: dict[str, Any]) -> None:
    if summary.get("run_id") != config.get("experiment_id"):
        fail("run id does not match config")
    inputs = summary.get("inputs")
    if not isinstance(inputs, dict):
        fail("summary inputs missing")
    if inputs.get("config_sha256") != sha256_file(config_path):
        fail("config hash mismatch")
    manifest_path = ROOT / str(config["dataset"]["manifest"])
    if inputs.get("manifest_sha256") != sha256_file(manifest_path):
        fail("manifest hash mismatch")


def validate_protocol(reference: dict[str, Any], config: dict[str, Any]) -> tuple[list[int], int]:
    evaluation = config["evaluation"]
    cutoffs = [int(value) for value in reference.get("cutoffs", [])]
    primary = int(reference.get("primary_cutoff", 0))
    if cutoffs != [int(value) for value in evaluation["cutoffs"]]:
        fail("reported cutoffs do not match the contract")
    if primary != int(evaluation["primary_cutoff"]) or primary != max(cutoffs):
        fail("primary cutoff must match the contract and be the deepest cutoff")
    propensity = reference.get("propensity")
    if not isinstance(propensity, dict):
        fail("propensity metadata missing")
    if float(propensity["gamma"]) != float(config["propensity"]["gamma"]):
        fail("propensity gamma does not match the contract")
    expected_max = 1.0 / float(config["propensity"]["minimum_propensity"])
    if finite_number(propensity.get("max_inverse_weight"), "max inverse weight") > expected_max + 1e-9:
        fail("inverse propensity weight exceeds the configured floor")
    return cutoffs, primary


def validate_reference_metrics(reference: dict[str, Any], cutoffs: list[int]) -> None:
    metrics = reference.get("full_reference_metrics")
    if not isinstance(metrics, dict) or set(metrics) != MODELS:
        fail("full-reference metrics missing a model")
    for model, per_cutoff in metrics.items():
        if set(per_cutoff) != {str(cutoff) for cutoff in cutoffs}:
            fail(f"{model} is missing a cutoff in the full reference")
        for cutoff, values in per_cutoff.items():
            for field in METRIC_FIELDS:
                finite_probability(values.get(field), f"{model}@{cutoff} {field}")
    strata = reference.get("full_reference_strata")
    if not isinstance(strata, dict) or set(strata) != MODELS:
        fail("full-reference strata missing a model")


def validate_controls(reference: dict[str, Any]) -> None:
    als = reference.get("als")
    if not isinstance(als, dict):
        fail("reference ALS control metadata missing")
    seeds = als.get("seeds")
    if not isinstance(seeds, list) or len(seeds) < 2 or len(set(seeds)) != len(seeds):
        fail("ALS control requires at least two distinct seeds")
    control = als.get("seed_variance_control")
    if not isinstance(control, dict):
        fail("ALS seed-variance control missing")
    pairs = control.get("pairs")
    expected_pairs = len(seeds) * (len(seeds) - 1) // 2
    if not isinstance(pairs, list) or len(pairs) != expected_pairs:
        fail(f"ALS seed-variance control needs {expected_pairs} seed pairs")
    observed_floor = 0.0
    for pair in pairs:
        value = pair.get("mean_absolute_ndcg_difference")
        finite_probability(value, f"ALS pair {pair.get('seeds')} mean absolute difference")
        observed_floor = max(observed_floor, float(value))
    if abs(float(control.get("max_mean_absolute_ndcg_difference", -1.0)) - observed_floor) > 1e-12:
        fail("ALS seed-variance floor does not match its pairs")
    item_item = reference.get("item_item")
    if not isinstance(item_item, dict) or int(item_item.get("supported_item_count", 0)) <= 0:
        fail("deterministic item-item probe metadata missing")
    ceilings = reference.get("evaluation_ceilings")
    if not isinstance(ceilings, dict):
        fail("evaluation ceiling audit missing")
    finite_probability(ceilings.get("share_with_relevant_above_cutoff"), "capped-user share")
    finite_probability(ceilings.get("mean_recall_ceiling"), "mean recall ceiling")


def validate_rows(
    rows: list[Any], reference: dict[str, Any], config: dict[str, Any],
    cutoffs: list[int], primary: int,
) -> None:
    caps = {int(value) for value in config.get("caps", [])}
    expected = {(policy, cap, model) for policy in POLICIES for cap in caps for model in MODELS}
    observed = {
        (str(row.get("policy")), int(row.get("cap")), str(row.get("model")))
        for row in rows if isinstance(row, dict)
    }
    if observed != expected or len(rows) != len(expected):
        fail(f"policy grid mismatch: {len(observed)} observed, {len(expected)} expected")

    pilot = int(reference.get("pilot_training_interactions", -1))
    cohort = int(reference.get("cohort_user_count", -1))
    if pilot <= 0 or cohort <= 0:
        fail("invalid fixed cohort or pilot interaction count")
    if int(reference.get("full_training_interactions", 0)) != pilot + int(
        reference.get("full_collection_interactions", -1)
    ):
        fail("full reference interaction invariant failed")
    if int(reference.get("evaluated_user_count", 0)) <= 0:
        fail("no future-test users were evaluated")

    for row in rows:
        cap = int(row["cap"])
        label = f"{row['policy']}/{cap}/{row['model']}"
        selected = int(row["selected_collection_interactions"])
        if selected < 0 or selected > cap * cohort:
            fail(f"cap invariant failed for {label}")
        if int(row["training_interactions"]) != pilot + selected:
            fail(f"training interaction invariant failed for {label}")
        if int(row["selected_collection_clients"]) > int(row["collection_clients"]):
            fail(f"client coverage invariant failed for {label}")
        if int(row["primary_cutoff"]) != primary:
            fail(f"row primary cutoff does not match the contract for {label}")
        by_cutoff = row.get("metrics_by_cutoff")
        if not isinstance(by_cutoff, dict) or set(by_cutoff) != {str(c) for c in cutoffs}:
            fail(f"{label} is missing a cutoff")
        for cutoff, values in by_cutoff.items():
            for field in METRIC_FIELDS:
                finite_probability(values.get(field), f"{label}@{cutoff} {field}")
        deltas = row.get("ndcg_delta_by_cutoff")
        if not isinstance(deltas, dict) or set(deltas) != {str(c) for c in cutoffs}:
            fail(f"{label} is missing a paired NDCG interval at some cutoff")
        for cutoff, entry in deltas.items():
            low = finite_number(entry.get("ci_low"), f"{label}@{cutoff} ci_low")
            high = finite_number(entry.get("ci_high"), f"{label}@{cutoff} ci_high")
            finite_number(entry.get("delta"), f"{label}@{cutoff} delta")
            if low > high:
                fail(f"invalid NDCG interval for {label}@{cutoff}")
        if abs(float(deltas[str(primary)]["delta"]) - float(row["mean_ndcg_delta_vs_full"])) > 1e-12:
            fail(f"{label} primary delta disagrees with its cutoff table")
        if abs(float(by_cutoff[str(primary)]["ndcg"]) - float(row["mean_ndcg"])) > 1e-12:
            fail(f"{label} primary NDCG disagrees with its cutoff table")
        for key in (
            "mean_ndcg_delta_vs_full", "mean_absolute_ndcg_error", "ndcg_delta_ci_low",
            "ndcg_delta_ci_high", "mean_weighted_ndcg_delta_vs_full",
            "tail_vs_equal_absolute_error_improvement",
            "tail_vs_equal_absolute_error_improvement_ci_low",
            "tail_vs_equal_absolute_error_improvement_ci_high",
        ):
            finite_number(row.get(key), f"{label} {key}")
        if float(row["ndcg_delta_ci_low"]) > float(row["ndcg_delta_ci_high"]):
            fail(f"invalid NDCG interval for {label}")
        strata = row.get("strata")
        if not isinstance(strata, dict) or not {"user_activity", "relevant_set_size", "hit_popularity_bands"} <= set(strata):
            fail(f"{label} is missing stratified reporting")
        covered = sum(entry["users"] for entry in strata["user_activity"].values())
        if covered != int(reference["evaluated_user_count"]):
            fail(f"{label} activity strata do not cover every evaluated user")


def validate(config_path: Path, result_root: Path) -> None:
    config = load_json(config_path)
    summary = load_json(result_root / "candidate_summary.json")
    reference = load_json(result_root / "reference_artifact.json")
    if summary.get("reference") != reference:
        fail("summary and reference artifact disagree")
    validate_inputs(config_path, summary, config)
    cutoffs, primary = validate_protocol(reference, config)
    validate_reference_metrics(reference, cutoffs)
    validate_controls(reference)
    rows = summary.get("rows")
    if not isinstance(rows, list):
        fail("summary rows missing")
    validate_rows(rows, reference, config, cutoffs, primary)

    script = summary.get("script")
    if not isinstance(script, dict):
        fail("summary script metadata missing")
    script_path = ROOT / str(script.get("path"))
    if not script_path.is_file() or script.get("sha256") != sha256_file(script_path):
        fail("runner hash mismatch")
    if not (result_root / "report.md").is_file():
        fail("report is missing")
    print(f"valid: {result_root}")
    print(
        f"valid: {len(rows)} policy/model rows, cutoffs {cutoffs}, primary {primary}, "
        f"{reference['evaluated_user_count']:,} evaluated users"
    )
    print("valid: propensity, seed-variance, ceiling, and strata controls present")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiments/fixed_cohort_budget_v2.json")
    )
    parser.add_argument(
        "--result-root", type=Path, default=Path("results/explorations/fixed_cohort_budget_v2")
    )
    args = parser.parse_args()
    validate(repo_path(args.config), repo_path(args.result_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
