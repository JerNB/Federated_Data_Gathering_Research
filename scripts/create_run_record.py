#!/usr/bin/env python3
"""Create a reproducible run record from a declared experiment configuration."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from validate_experiment import ROOT, load_json, sha256_file, validate_experiment


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


def read_json_object(path: Path) -> dict[str, Any]:
    value = load_json(path)
    return value


def read_json_value(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        fail(f"cannot read {path}: {exc}")
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON in {path}: {exc}")


def git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        fail(f"cannot determine the current Git commit: {exc}")
    return result.stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/milestone_1_oracle_comparison.json"))
    parser.add_argument("--manifest", type=Path, default=Path("data/dataset_manifest.json"))
    parser.add_argument("--variant", required=True, help="model variant identifier from the experiment config")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status", choices=["planned", "running", "completed", "failed"], default="planned")
    parser.add_argument("--metrics", type=Path, help="JSON object containing completed run metrics")
    parser.add_argument("--support-report", type=Path, help="JSON object containing the support report")
    parser.add_argument("--artifacts", type=Path, help="JSON array containing artifact records")
    parser.add_argument("--notes", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = ROOT / args.config
    manifest_path = ROOT / args.manifest
    output_path = ROOT / args.output
    if output_path.exists() and not args.force:
        fail(f"output exists: {output_path}; use --force to replace it")

    validate_experiment(config_path, manifest_path)
    config = load_json(config_path)
    manifest = load_json(manifest_path)
    variant = next(
        (item for item in config["model_variants"] if item["variant_id"] == args.variant),
        None,
    )
    if variant is None:
        fail(f"unknown model variant: {args.variant}")

    objective = load_json(ROOT / config["objective"]["path"])
    metrics: dict[str, Any] = {}
    if args.metrics:
        metrics = read_json_object(ROOT / args.metrics)
        if not all(isinstance(value, (int, float)) for value in metrics.values()):
            fail("metrics must contain only numeric values")
    support_report: dict[str, Any] = {
        "support_check_passed": False,
        "eligible_item_count": 0,
        "per_cluster": [],
    }
    if args.support_report:
        support_report = read_json_object(ROOT / args.support_report)
    artifacts: list[Any] = []
    if args.artifacts:
        value = read_json_value(ROOT / args.artifacts)
        if not isinstance(value, list):
            fail("artifacts must be a JSON array")
        artifacts = value
    if args.status == "completed" and not metrics:
        fail("completed runs require a metrics JSON object")
    if args.status == "completed" and support_report.get("support_check_passed") is not True:
        fail("completed runs require a passed support report")

    record = {
        "run_id": args.run_id,
        "experiment_id": config["experiment_id"],
        "status": args.status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": git_commit(),
        "dataset": {
            "manifest_path": str(args.manifest),
            "dataset_id": manifest["dataset_id"],
            "version": manifest["version"],
            "manifest_sha256": sha256_file(manifest_path),
        },
        "configuration": {
            "path": str(args.config),
            "sha256": sha256_file(config_path),
            "variant_id": args.variant,
        },
        "formula": {
            "objective_id": objective["objective_id"],
            "score": variant["score_formula"],
            "loss": objective["loss"],
        },
        "split": {
            "method": config["split"]["method"],
            "seed": config["pilot"]["user_selection"]["seed"],
            "train_fraction": config["split"]["train_fraction"],
            "validation_fraction": config["split"]["validation_fraction"],
            "test_fraction": config["split"]["test_fraction"],
            "user_count": config["pilot"]["user_selection"]["target_count"],
        },
        "support_report": support_report,
        "metrics": metrics,
        "artifacts": artifacts,
        "notes": args.notes,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"created {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
