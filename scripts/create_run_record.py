#!/usr/bin/env python3
"""Create a reproducible run record and append its summary to the registry."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from validate_experiment import ROOT, load_json, sha256_file, validate_experiment

LIFECYCLE_TRANSITIONS: dict[str, set[str]] = {
    "planned": {"running"},
    "running": {"completed", "failed"},
    "completed": set(),
    "failed": set(),
}


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

def schema_errors(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    errors: list[str] = []
    expected = schema.get("type")
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: expected constant {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: value {value!r} is not in the allowed enum")
    if expected == "object" and not isinstance(value, dict):
        return errors + [f"{path}: expected object"]
    if expected == "array" and not isinstance(value, list):
        return errors + [f"{path}: expected array"]
    if expected == "string" and not isinstance(value, str):
        return errors + [f"{path}: expected string"]
    if expected == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
        return errors + [f"{path}: expected integer"]
    if expected == "number" and (not isinstance(value, (int, float)) or isinstance(value, bool)):
        return errors + [f"{path}: expected number"]
    if expected == "boolean" and not isinstance(value, bool):
        return errors + [f"{path}: expected boolean"]

    if isinstance(value, dict):
        properties = schema.get("properties", {})
        for required in schema.get("required", []):
            if required not in value:
                errors.append(f"{path}: missing required property {required!r}")
        additional = schema.get("additionalProperties", True)
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key in properties:
                errors.extend(schema_errors(child, properties[key], child_path))
            elif additional is False:
                errors.append(f"{child_path}: additional property is not allowed")
            elif isinstance(additional, dict):
                errors.extend(schema_errors(child, additional, child_path))
        if "minProperties" in schema and len(value) < schema["minProperties"]:
            errors.append(f"{path}: expected at least {schema['minProperties']} properties")
    if isinstance(value, list):
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                errors.extend(schema_errors(item, item_schema, f"{path}[{index}]"))
    if isinstance(value, str) and "pattern" in schema:
        if re.search(schema["pattern"], value) is None:
            errors.append(f"{path}: value does not match the required pattern")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and "minimum" in schema:
        if value < schema["minimum"]:
            errors.append(f"{path}: value is below the minimum")
    for branch in schema.get("allOf", []):
        condition = branch.get("if")
        if isinstance(condition, dict) and not schema_errors(value, condition, path):
            errors.extend(schema_errors(value, branch.get("then", {}), path))
    return errors


def validate_record_schema(record: dict[str, Any]) -> None:
    schema_path = ROOT / "experiments" / "run_record.schema.json"
    schema = load_json(schema_path)
    errors = schema_errors(record, schema)
    if errors:
        fail(
            f"run record does not match {schema_path}:\n"
            + "\n".join(f"  {error}" for error in errors)
        )


def read_support_report(path: Path) -> dict[str, Any]:
    payload = read_json_object(path)
    if "support_report" not in payload:
        return payload
    support_report = payload["support_report"]
    if not isinstance(support_report, dict):
        fail(f"support_report in {path} must be an object")
    return support_report


def read_registry(path: Path) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return entries
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            summary = json.loads(line)
        except json.JSONDecodeError as exc:
            fail(f"invalid registry JSON on line {line_number}: {exc}")
        if not isinstance(summary, dict):
            fail(f"registry line {line_number} must contain an object")
        for field in ("run_id", "status", "created_at_utc", "record_path"):
            if not summary.get(field):
                fail(f"registry line {line_number} must contain {field}")
        entries[summary["run_id"]] = summary
    return entries


def append_registry(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    read_registry(path)
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
    parser.add_argument("--support-report", type=Path, help="JSON object or selector output containing the support report")
    parser.add_argument("--artifacts", type=Path, help="JSON array containing artifact records")
    parser.add_argument("--notes", action="append", default=[])
    parser.add_argument("--force", action="store_true", help="allow a valid lifecycle transition for an existing run")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = repo_path(args.config)
    manifest_path = repo_path(args.manifest)
    output_path = repo_path(args.output) if args.output else ROOT / "results" / "run_records" / f"{args.run_id}.json"
    registry_path = repo_path(args.registry)
    registry_entries = read_registry(registry_path)
    previous_summary = registry_entries.get(args.run_id)
    existing_record = read_json_object(output_path) if output_path.exists() else None
    if existing_record is not None:
        if not args.force:
            fail(f"output exists: {output_path}; use --force for a lifecycle transition")
        if existing_record.get("run_id") != args.run_id:
            fail(f"output belongs to a different run_id: {output_path}")
    if previous_summary is None:
        if existing_record is not None:
            fail(f"output exists without a registry entry: {output_path}; repair the registry or choose a new output")
    else:
        if not args.force:
            fail(f"run_id already exists in registry: {args.run_id}; use --force for a lifecycle transition")
        if existing_record is None:
            fail(f"registry entry has no full record: {previous_summary['record_path']}")
        if previous_summary["record_path"] != display_path(output_path):
            fail(
                f"run_id {args.run_id} must reuse record path {previous_summary['record_path']}"
            )
        if existing_record.get("created_at_utc") != previous_summary["created_at_utc"]:
            fail(f"full record timestamp disagrees with registry for {args.run_id}")
        if existing_record.get("status") != previous_summary["status"]:
            fail(f"full record status disagrees with registry for {args.run_id}")
        previous_status = previous_summary["status"]
        allowed_statuses = LIFECYCLE_TRANSITIONS.get(previous_status, set())
        if args.status not in allowed_statuses:
            fail(
                f"invalid lifecycle transition for {args.run_id}: "
                f"{previous_status} -> {args.status}"
            )

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
        if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in metrics.values()):
            fail("metrics must contain only numeric values")
    support_report: dict[str, Any] = {
        "support_check_passed": False,
        "eligible_item_count_by_cluster": {},
        "eligible_item_intersection_count": 0,
        "per_cluster": [],
    }
    if args.support_report:
        support_report = read_support_report(repo_path(args.support_report))
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

    current_at = datetime.now(timezone.utc).isoformat()
    created_at = previous_summary["created_at_utc"] if previous_summary else current_at
    updated_at = current_at if previous_summary else None
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
    if updated_at is not None:
        record["updated_at_utc"] = updated_at
    validate_record_schema(record)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    summary = {
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
    }
    if updated_at is not None:
        summary["updated_at_utc"] = updated_at
    append_registry(registry_path, summary)
    print(f"created {output_path}")
    print(f"appended {registry_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
