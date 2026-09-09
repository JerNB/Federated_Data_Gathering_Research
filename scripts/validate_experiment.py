#!/usr/bin/env python3
"""Validate the dataset manifest and the first experiment contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        fail(f"cannot read {path}: {exc}")
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON in {path}: {exc}")
    if not isinstance(value, dict):
        fail(f"top level of {path} must be an object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_checksum_manifest(path: Path) -> dict[str, str]:
    if not path.is_file():
        fail(f"checksum manifest not found: {path}")
    checksums: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            fail(f"invalid checksum line {line_number} in {path}")
        checksum, relative_path = parts
        relative_path = relative_path.lstrip(" *")
        if len(checksum) != 64:
            fail(f"invalid SHA 256 value on line {line_number} in {path}")
        checksums[relative_path] = checksum
    return checksums


def validate_manifest(manifest_path: Path, dataset_root: Path | None, verify_files: bool) -> None:
    manifest = load_json(manifest_path)
    required = {"dataset_id", "version", "source", "summary", "archive", "files", "integrity"}
    missing = required.difference(manifest)
    if missing:
        fail(f"manifest is missing fields: {sorted(missing)}")
    if manifest["dataset_id"] != "movielens_ml_latest":
        fail("manifest dataset_id must be movielens_ml_latest")
    if manifest["version"] != "2023-07-20":
        fail("manifest version must be 2023-07-20")
    if manifest["archive"]["sha256"] != "21c09ce12e8062c6237011432fbd3acacb9e80d094f8400b1ce8c7592f725804":
        fail("manifest archive checksum does not match the pinned release")

    checksum_path = ROOT / manifest["integrity"]["file_manifest"]
    checksum_map = read_checksum_manifest(checksum_path)
    files = manifest["files"]
    if not isinstance(files, list) or not files:
        fail("manifest files must be a non empty list")
    for item in files:
        if not isinstance(item, dict) or not {"path", "sha256"}.issubset(item):
            fail("each manifest file entry needs path and sha256")
        relative_path = item["path"]
        if checksum_map.get(relative_path) != item["sha256"]:
            fail(f"manifest checksum mismatch for {relative_path}")
        if dataset_root is not None:
            source_path = dataset_root.parent / relative_path
            if not source_path.is_file():
                fail(f"source file not found: {source_path}")
            if verify_files:
                actual = sha256_file(source_path)
                if actual != item["sha256"]:
                    fail(f"source checksum mismatch for {source_path}")

    print(f"valid: dataset manifest {manifest_path}")
    if dataset_root is not None:
        mode = "with file checksums" if verify_files else "with file presence"
        print(f"valid: source snapshot {dataset_root} {mode}")


def validate_experiment(config_path: Path, manifest_path: Path) -> None:
    config = load_json(config_path)
    manifest = load_json(manifest_path)
    required = {"experiment_id", "dataset", "objective", "split", "pilot", "model_variants", "capacity", "metrics", "interpretation_gate"}
    missing = required.difference(config)
    if missing:
        fail(f"experiment config is missing fields: {sorted(missing)}")

    dataset = config["dataset"]
    if dataset["dataset_id"] != manifest["dataset_id"]:
        fail("experiment dataset_id does not match the manifest")
    if dataset["version"] != manifest["version"]:
        fail("experiment dataset version does not match the manifest")
    objective_path = ROOT / config["objective"]["path"]
    objective = load_json(objective_path)
    if objective["objective_id"] != config["objective"]["objective_id"]:
        fail("experiment objective_id does not match the objective file")

    split = config["split"]
    fractions = [split["train_fraction"], split["validation_fraction"], split["test_fraction"]]
    if abs(sum(fractions) - 1.0) > 1e-9:
        fail("split fractions must sum to one")
    if split["cluster_fit_source"] != "training_only" or split["catalog_fit_source"] != "training_only":
        fail("cluster and catalog construction must use training only data")

    pilot = config["pilot"]
    user_selection = pilot["user_selection"]
    if user_selection["target_count"] != 5000:
        fail("the pilot must use the fixed 5000 user target")
    if user_selection["eligibility_source"] != "pilot_train_only":
        fail("pilot user eligibility must use pilot training data")
    support = pilot["support_filter"]
    if support["source"] != "pilot_train_only":
        fail("support filtering must use pilot training data")
    selector_path = ROOT / support["selector"]
    if not selector_path.is_file():
        fail(f"support selector not found: {selector_path}")
    if support["minimum_interactions_per_item_per_cluster"] <= 0:
        fail("support threshold must be positive")
    if support["minimum_eligible_items"] <= 0:
        fail("minimum eligible item count must be positive")
    if not support["required_report"]:
        fail("support report fields are required")
    evaluation = pilot["evaluation"]
    if evaluation["sweep_catalog"] != "support_filtered_training_catalog":
        fail("the pilot sweep must use the support filtered training catalog")
    if evaluation["same_catalog_for_all_variants"] is not True:
        fail("all pilot model variants must use the same catalog")
    confirmation = evaluation["full_snapshot_confirmation"]
    if confirmation["catalog"] != "all_declared_items":
        fail("final confirmation must use the full declared catalog")

    variants = config["model_variants"]
    expected = {"global_mf", "oracle_clustered_mf"}
    actual = {variant.get("variant_id") for variant in variants}
    if len(variants) != 2 or actual != expected:
        fail("milestone 1 must contain exactly global_mf and oracle_clustered_mf")
    global_variant = next(variant for variant in variants if variant["variant_id"] == "global_mf")
    oracle_variant = next(variant for variant in variants if variant["variant_id"] == "oracle_clustered_mf")
    if global_variant["routing"] != "none":
        fail("global_mf must have no routing")
    if oracle_variant["routing"] != "oracle_assignment":
        fail("oracle_clustered_mf must use declared oracle assignment")
    if config["capacity"]["primary_policy"] != "equal_total_trainable_parameter_budget":
        fail("the primary capacity policy must match total trainable parameters")
    if not config["metrics"]:
        fail("at least one metric is required")

    print(f"valid: experiment config {config_path}")
    print(f"valid: objective {objective_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/milestone_1_oracle_comparison.json"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/dataset_manifest.json"),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="optional raw snapshot directory to check",
    )
    parser.add_argument(
        "--verify-files",
        action="store_true",
        help="hash every raw file listed in the manifest",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = ROOT / args.manifest
    config_path = ROOT / args.config
    dataset_root = ROOT / args.dataset_root if args.dataset_root else None
    if args.verify_files and dataset_root is None:
        fail("--verify-files requires --dataset-root")
    validate_manifest(manifest_path, dataset_root, args.verify_files)
    validate_experiment(config_path, manifest_path)
    print("validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
