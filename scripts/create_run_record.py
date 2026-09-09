#!/usr/bin/env python3
"""Create a reproducible run record and append its summary to the registry."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from validate_experiment import ROOT, load_json, sha256_file, validate_experiment


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


def repo_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json_object(path: Path) -> dict[str, Any]:
    return load_json(path)


def read_json_value(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        fail(f"cannot read {path}: {exc}")
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON in {path}: {exc}")


def git_state() -> tuple[str, list[str]]:
    try:
        commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        status_result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        fail(f"cannot determine Git state: {exc}")
    return commit_result.stdout.strip(), status_result.stdout.splitlines()


def append_registry(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                previous = json.loads(line)
            except json.JSONDecodeError as exc:
                fail(f"invalid registry JSON on line {line_number}: {exc}")
            if previous.get("run_id") == summary["run_id"]:
                fail(f"run_id already exists in registry: {summary['run_id']}")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/milestone_1_oracle_comparison.json"))
    parser.add_argument("--manifest", type=Path, default=Path("data/dataset_manifest.json"))
    parser.add_argument("--variant", required=True, help="model variant identifier from the experiment config")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, help="full run record path; defaults to results/run_records/<run_id>.json")
    parser.add_argument("--registry", type=Path, default=Path("results/run_registry.jsonl"))
    parser.add_argument("--status", choices=["planned", "running", "completed", "failed"], default="planned")
    parser.add_argument("--metrics", type=Path, help="JSON object containing completed run metrics")
    parser.add_argument("--support-report", type=Path, help="JSON object containing the support report")
    parser.add_argument("--artifacts", type=Path, help="JSON array containing artifact records")
    parser.add_argument("--notes", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = repo_path(args.config)
    manifest_path = repo_path(args.manifest)
    output_path = repo_path(args.output) if args.output else ROOT / "results" / "run_records" / f"{args.run_id}.json"
    registry_path = repo_path(args.registry)
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

    objective = load_json(repo_path(Path(config["objective"]["path"])))
    metrics: dict[str, Any] = {}
    if args.metrics:
        metrics = read_json_object(repo_path(args.metrics))
        if not all(isinstance(value, (int, float)) for value in metrics.values()):
            fail("metrics must contain only numeric values")
    support_report: dict[str, Any] = {
        "support_check_passed": False,
        "eligible_item_count_by_cluster": {},
        "eligible_item_intersection_count": 0,
        "per_cluster": [],
    }
    if args.support_report:
        support_report = read_json_object(repo_path(args.support_report))
    artifacts: list[Any] = []
    if args.artifacts:
        value = read_json_value(repo_path(args.artifacts))
        if not isinstance(value, list):
            fail("artifacts must be a JSON array")
        artifacts = value

    code_commit, worktree_status = git_state()
    worktree_clean = not worktree_status
    if args.status == "completed" and not metrics:
        fail("completed runs require a metrics JSON object")
    if args.status == "completed" and support_report.get("support_check_passed") is not True:
        fail("completed runs require a passed support report")
    if args.status == "completed" and not worktree_clean:
        fail("completed runs require a clean worktree")

    created_at = datetime.now(timezone.utc).isoformat()
    record = {
        "run_id": args.run_id,
        "experiment_id": config["experiment_id"],
        "status": args.status,
        "created_at_utc": created_at,
        "code_commit": code_commit,
        "worktree_clean": worktree_clean,
        "worktree_status": worktree_status,
        "dataset": {
            "manifest_path": display_path(manifest_path),
            "dataset_id": manifest["dataset_id"],
            "version": manifest["version"],
            "manifest_sha256": sha256_file(manifest_path),
        },
        "configuration": {
            "path": display_path(config_path),
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
    append_registry(
        registry_path,
        {
            "run_id": args.run_id,
            "experiment_id": config["experiment_id"],
            "status": args.status,
            "created_at_utc": created_at,
            "code_commit": code_commit,
            "worktree_clean": worktree_clean,
            "dataset_id": manifest["dataset_id"],
            "dataset_version": manifest["version"],
            "variant_id": args.variant,
            "metrics": metrics,
            "record_path": display_path(output_path),
        },
    )
    print(f"created {output_path}")
    print(f"appended {registry_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
