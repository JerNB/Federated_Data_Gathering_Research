#!/usr/bin/env python3
"""Validate fixed-cohort local-data-budget replay artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
POLICIES = {"equal_chronological_cap", "tail_reserve_cap"}
MODELS = {"popularity", "rating_weighted_popularity"}


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


def validate(config_path: Path, result_root: Path) -> None:
    config = load_json(config_path)
    summary = load_json(result_root / "candidate_summary.json")
    reference = load_json(result_root / "reference_artifact.json")
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
    if summary.get("reference") != reference:
        fail("summary and reference artifact disagree")

    caps = {int(value) for value in config.get("caps", [])}
    models = {str(value) for value in config.get("evaluation", {}).get("models", [])}
    if models != MODELS:
        fail(f"unexpected configured models: {models}")
    rows = summary.get("rows")
    if not isinstance(rows, list):
        fail("summary rows missing")
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
    if int(reference.get("full_training_interactions", 0)) != pilot + int(reference.get("full_collection_interactions", -1)):
        fail("full reference interaction invariant failed")
    if int(reference.get("evaluated_user_count", 0)) <= 0:
        fail("no future-test users were evaluated")

    for row in rows:
        if not isinstance(row, dict):
            fail("non-object row")
        cap = int(row["cap"])
        selected = int(row["selected_collection_interactions"])
        training = int(row["training_interactions"])
        if selected < 0 or selected > cap * cohort:
            fail(f"cap invariant failed for {row['policy']}/{cap}")
        if training != pilot + selected:
            fail(f"training interaction invariant failed for {row['policy']}/{cap}")
        if int(row["selected_collection_clients"]) > int(row["collection_clients"]):
            fail(f"client coverage invariant failed for {row['policy']}/{cap}")
        if int(row["collection_clients"]) > cohort:
            fail(f"collection client count exceeds cohort for {row['policy']}/{cap}")
        for key in ("mean_ndcg_at_10", "mean_recall_at_10", "mean_absolute_ndcg_error", "selected_tail_share"):
            finite_probability(row.get(key), f"{row['policy']}/{cap}/{row['model']} {key}")
        for key in ("ndcg_delta_ci_low", "ndcg_delta_ci_high", "mean_ndcg_delta_vs_full", "mean_recall_delta_vs_full", "tail_vs_equal_absolute_error_improvement", "tail_vs_equal_absolute_error_improvement_ci_low", "tail_vs_equal_absolute_error_improvement_ci_high"):
            value = row.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                fail(f"{row['policy']}/{cap}/{row['model']} {key} is not finite")
        if float(row["ndcg_delta_ci_low"]) > float(row["ndcg_delta_ci_high"]):
            fail(f"invalid NDCG interval for {row['policy']}/{cap}/{row['model']}")
        if float(row["tail_vs_equal_absolute_error_improvement_ci_low"]) > float(row["tail_vs_equal_absolute_error_improvement_ci_high"]):
            fail(f"invalid tail-comparison interval for {row['policy']}/{cap}/{row['model']}")

    script = summary.get("script")
    if not isinstance(script, dict):
        fail("summary script metadata missing")
    script_path = ROOT / str(script.get("path"))
    if not script_path.is_file() or script.get("sha256") != sha256_file(script_path):
        fail("runner hash mismatch")
    if not (result_root / "report.md").is_file():
        fail("report is missing")
    print(f"valid: {result_root}")
    print(f"valid: {len(rows)} policy/model rows across {len(caps)} caps")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/fixed_cohort_budget_v1.json"))
    parser.add_argument("--result-root", type=Path, default=Path("results/explorations/fixed_cohort_budget_v1"))
    args = parser.parse_args()
    validate(repo_path(args.config), repo_path(args.result_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
