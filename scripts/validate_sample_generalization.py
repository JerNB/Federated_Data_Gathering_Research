#!/usr/bin/env python3
"""Validate the executed sample-generalization contract and artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SCHEMES = {
    "uniform_user",
    "activity_stratified_user",
    "uniform_interaction",
    "within_user_history",
}
EXPECTED_MODELS = {
    "popularity",
    "rating_weighted_popularity",
    "item_item_cosine",
}
EXPECTED_FRACTIONS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read valid JSON from {path}: {exc}")
    if not isinstance(payload, dict):
        fail(f"top level of {path} must be an object")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(actual: float, expected: float, tolerance: float = 1e-12) -> bool:
    return abs(actual - expected) <= tolerance


def validate(config_path: Path, result_root: Path) -> None:
    config = load_json(config_path)
    summary = load_json(result_root / "candidate_summary.json")
    reference = load_json(result_root / "reference_artifact.json")

    if config.get("experiment_id") != "sample_generalization_v1":
        fail("unexpected experiment id")
    config_dataset = config.get("dataset", {})
    if config_dataset.get("dataset_id") != "movielens_ml_latest" or config_dataset.get("version") != "2023-07-20":
        fail("config dataset is not the pinned MovieLens snapshot")
    configured_primary_schemes = {
        item.get("id")
        for item in config.get("sampling_candidates", [])
        if item.get("role") == "primary"
    }
    configured_models = {
        item.get("id")
        for item in config.get("model_candidates", [])
        if item.get("implemented")
    }
    if configured_primary_schemes != EXPECTED_SCHEMES:
        fail("config primary sampling candidates differ from the executed matrix")
    if configured_models != EXPECTED_MODELS:
        fail("config implemented model candidates differ from the executed matrix")
    candidate_ids = {item.get("id") for item in config.get("sampling_candidates", [])}
    if not {"item_stratified", "recent_window", "cluster_stratified_user"}.issubset(candidate_ids):
        fail("config omits documented secondary sampling areas")
    if config.get("constant_controls", {}).get("support_threshold") != 20:
        fail("config support threshold is not fixed at 20")
    if config.get("constant_controls", {}).get("cutoff") != 10:
        fail("config cutoff is not fixed at 10")
    if config.get("fractions") != list(EXPECTED_FRACTIONS):
        fail("config fractions differ from the executed ladder")
    if summary.get("inputs", {}).get("manifest_sha256") != config_dataset.get("manifest_sha256"):
        fail("summary and config manifest digests differ")

    design = summary.get("design", {})
    schemes = set(design.get("schemes", []))
    models = set(design.get("models", []))
    fractions = tuple(float(value) for value in design.get("fractions", []))
    if schemes != EXPECTED_SCHEMES:
        fail(f"executed schemes differ: {sorted(schemes)}")
    if models != EXPECTED_MODELS:
        fail(f"executed models differ: {sorted(models)}")
    if fractions != EXPECTED_FRACTIONS:
        fail(f"executed fractions differ: {fractions}")
    if design.get("replicates") != 10 or design.get("panel_size") != 2000:
        fail("replicate or panel controls differ from the full contract")

    dataset = summary.get("dataset", {})
    expected_dataset = {
        "raw_rows": 33832162,
        "user_count": 330975,
        "item_count": 83239,
        "positive_training_rows": 13653758,
        "panel_user_count": 2000,
    }
    for key, expected in expected_dataset.items():
        if dataset.get(key) != expected:
            fail(f"dataset {key}={dataset.get(key)!r}, expected {expected!r}")

    rows = summary.get("rows")
    aggregates = summary.get("aggregates")
    if not isinstance(rows, list) or len(rows) != 732:
        fail(f"expected 732 draw rows, got {len(rows) if isinstance(rows, list) else type(rows).__name__}")
    if not isinstance(aggregates, list) or len(aggregates) != 84:
        fail(f"expected 84 aggregate rows, got {len(aggregates) if isinstance(aggregates, list) else type(aggregates).__name__}")

    full_rows = [row for row in rows if close(float(row.get("fraction", -1)), 1.0)]
    if len(full_rows) != 12:
        fail(f"expected 12 reused full-fraction rows, got {len(full_rows)}")
    if any(float(row.get("absolute_ndcg_error", 1.0)) != 0.0 for row in full_rows):
        fail("a full-fraction row has nonzero absolute NDCG error")
    if any(int(row.get("sample_interactions", -1)) != 13653758 for row in full_rows):
        fail("a full-fraction row does not use the full positive training count")

    expected_result_root = config["executed_matrix"]["result_root"]
    if result_root.relative_to(ROOT).as_posix() != expected_result_root:
        fail("result root differs from the executable contract")
    runner = ROOT / "scripts/run_sample_generalization.py"
    if summary.get("script", {}).get("sha256") != sha256_file(runner):
        fail("result script hash does not match the checked-out runner")
    if reference.get("reference_id") != summary.get("inputs", {}).get("reference_cache_key"):
        fail("reference artifact and summary cache identities differ")
    if reference.get("dataset", {}).get("raw_rows") != 33832162:
        fail("reference artifact dataset count is not pinned")

    expected_figures = {
        "metric_vs_fraction.png",
        "relative_error_vs_fraction.png",
        "coverage_vs_fraction.png",
        "algorithm_order_agreement.png",
        "native_frame_gap_vs_fraction.png",
        "cost_vs_error.png",
    }
    figures = {Path(path).name for path in summary.get("figures", [])}
    if figures != expected_figures:
        fail(f"figure set differs: {sorted(figures)}")
    missing = [name for name in expected_figures if not (result_root / "figures" / name).is_file()]
    if missing:
        fail(f"missing figure files: {missing}")
    for required in ("report.md", "candidate_summary.json", "reference_artifact.json"):
        if not (result_root / required).is_file():
            fail(f"missing required artifact: {result_root / required}")

    print(f"valid: {result_root}")
    print("valid: 732 draw rows, 84 aggregate rows, 12 exact full-reference rows")
    print("valid: six figures, report, reference artifact, and runner hash")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/sample_generalization_v1.json"))
    parser.add_argument("--result-root", type=Path, default=Path("results/explorations/sample_generalization_full"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate(ROOT / args.config, ROOT / args.result_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
