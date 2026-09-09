#!/usr/bin/env python3
"""Smoke-test selector composition, schema rejection, and run lifecycle rules."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from create_run_record import ROOT, load_json, schema_errors, schema_keyword_errors


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True)


def assert_success(result: subprocess.CompletedProcess[str], label: str) -> None:
    assert result.returncode == 0, f"{label} failed:\n{result.stdout}\n{result.stderr}"


def assert_failure(result: subprocess.CompletedProcess[str], text: str, label: str) -> None:
    output = result.stdout + result.stderr
    assert result.returncode != 0 and text in output, f"{label} unexpectedly passed:\n{output}"


def main() -> int:
    selector = ROOT / "scripts" / "select_pilot_catalog.py"
    recorder = ROOT / "scripts" / "create_run_record.py"
    schema = load_json(ROOT / "experiments" / "run_record.schema.json")
    assert not schema_keyword_errors(schema), "run record schema uses unsupported keywords"
    assert schema_keyword_errors({"unsupported_keyword": True})

    with tempfile.TemporaryDirectory(prefix="fdg-selftest-") as directory:
        temporary = Path(directory)
        interactions = temporary / "interactions.csv"
        clusters = temporary / "clusters.csv"
        pilot = temporary / "pilot.json"
        record_path = temporary / "record.json"
        registry = temporary / "registry.jsonl"
        metrics = temporary / "metrics.json"
        interactions.write_text("user_id,item_id\n1,10\n2,10\n3,20\n4,20\n", encoding="utf-8")
        clusters.write_text("user_id,cluster_id\n1,0\n2,0\n3,1\n4,1\n", encoding="utf-8")
        metrics.write_text('{"loss": 1.0}\n', encoding="utf-8")

        result = run(
            [
                sys.executable,
                str(selector),
                "--interactions",
                str(interactions),
                "--cluster-map",
                str(clusters),
                "--output",
                str(pilot),
                "--users",
                "4",
                "--clusters",
                "2",
                "--seed",
                "1",
                "--min-user-interactions",
                "1",
                "--min-item-support-per-cluster",
                "2",
                "--min-eligible-items-per-cluster",
                "1",
            ]
        )
        assert_success(result, "selector")
        selector_document = json.loads(pilot.read_text(encoding="utf-8"))
        assert schema_errors(selector_document, schema), "raw selector output was accepted as a run record"

        result = run(
            [
                sys.executable,
                str(recorder),
                "--variant",
                "global_mf",
                "--run-id",
                "selftest_run",
                "--output",
                str(record_path),
                "--registry",
                str(registry),
                "--support-report",
                str(pilot),
                "--status",
                "planned",
            ]
        )
        assert_success(result, "planned record")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        assert record["support_report"] == selector_document["support_report"]
        assert not schema_errors(record, schema), "valid assembled record was rejected"

        invalid_record = dict(record)
        invalid_record["bogus_extra"] = True
        assert any("bogus_extra" in error for error in schema_errors(invalid_record, schema))
        assert any("at least 1 properties" in error for error in schema_errors({"status": "completed", "metrics": {}}, schema))

        result = run(
            [
                sys.executable,
                str(recorder),
                "--variant",
                "global_mf",
                "--run-id",
                "selftest_run",
                "--output",
                str(record_path),
                "--registry",
                str(registry),
                "--support-report",
                str(pilot),
                "--metrics",
                str(metrics),
                "--status",
                "completed",
                "--force",
            ]
        )
        assert_failure(result, "invalid lifecycle transition", "planned to completed transition")

        result = run(
            [
                sys.executable,
                str(recorder),
                "--variant",
                "oracle_clustered_mf",
                "--run-id",
                "selftest_run",
                "--output",
                str(record_path),
                "--registry",
                str(registry),
                "--support-report",
                str(pilot),
                "--status",
                "running",
                "--force",
            ]
        )
        assert_failure(result, "run identity mismatch", "variant identity transition")

        result = run(
            [
                sys.executable,
                str(recorder),
                "--variant",
                "global_mf",
                "--run-id",
                "selftest_run",
                "--output",
                str(record_path),
                "--registry",
                str(registry),
                "--support-report",
                str(pilot),
                "--status",
                "running",
                "--force",
            ]
        )
        assert_success(result, "running transition")
        result = run(
            [
                sys.executable,
                str(recorder),
                "--variant",
                "global_mf",
                "--run-id",
                "selftest_run",
                "--output",
                str(record_path),
                "--registry",
                str(registry),
                "--support-report",
                str(pilot),
                "--status",
                "failed",
                "--force",
            ]
        )
        assert_success(result, "failed transition")
        final_record = json.loads(record_path.read_text(encoding="utf-8"))
        registry_rows = [json.loads(line) for line in registry.read_text(encoding="utf-8").splitlines()]
        assert [row["status"] for row in registry_rows] == ["planned", "running", "failed"]
        assert final_record["created_at_utc"] == registry_rows[0]["created_at_utc"]
        assert final_record["updated_at_utc"]

    print("self-test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
