#!/usr/bin/env python3
"""Serve the research dashboard and expose tracked research metadata."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import subprocess
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "web"
EXPLORATION_ROOT = ROOT / "results" / "explorations"
RUN_ROOT = ROOT / "results" / "run_records"
ARTIFACT_ROOT = ROOT / "artifacts"
BOARD_PATH = ROOT / "results" / "research_board.json"
PROPOSAL_PATH = ROOT / "docs" / "research_proposal.md"
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")
BOARD_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{1,63}$")
BOARD_STATUSES = {"todo", "doing", "done", "blocked"}
BOARD_PRIORITIES = {"low", "normal", "high"}

JOB_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{1,95}$")


class BoardConflictError(ValueError):
    """A board command conflicts with an active job."""


ACTIVE_JOBS: dict[str, dict] = {}

BOARD_COMMANDS = {
    "make validate-experiment": {"target": "validate-experiment", "parameters": set()},
    "make explore": {"target": "explore", "parameters": {"RUN_ID", "SUPPORT_PROBE_USERS"}},
    "create_run_record.py": {
        "args": ["scripts/create_run_record.py"],
        "parameters": {"--variant", "--run-id", "--status"},
    },
}
BOARD_LOCK = Lock()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_child(root: Path, relative: str) -> Path | None:
    candidate = (root / relative).resolve()
    root_resolved = root.resolve()
    if candidate == root_resolved or root_resolved not in candidate.parents:
        return None
    return candidate

def default_board() -> dict:
    return {"version": 1, "items": []}


def load_board() -> dict:
    if not BOARD_PATH.is_file():
        return default_board()
    payload = load_json(BOARD_PATH)
    if not isinstance(payload.get("items"), list):
        raise ValueError("research board items must be a list")
    return payload


def save_board(board: dict) -> None:
    temporary = BOARD_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(board, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, BOARD_PATH)


def board_snapshot() -> dict:
    with BOARD_LOCK:
        try:
            return load_board()
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return default_board()


def board_item_id(title: str, existing: list[dict]) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:56] or "item"
    candidate = base
    suffix = 2
    used = {item.get("id") for item in existing}
    while candidate in used:
        candidate = f"{base}-{suffix}"
        suffix += 1
    return candidate


def clean_board_fields(payload: dict, current: dict | None = None) -> dict:
    source = {**(current or {}), **payload}
    title = str(source.get("title", "")).strip()
    status = source.get("status", "todo")
    owner = str(source.get("owner", "")).strip()
    priority = source.get("priority", "normal")
    notes = str(source.get("notes", "")).strip()
    command = source.get("command", "make explore")
    parameters = source.get("parameters")
    if parameters is None:
        parameters = {"RUN_ID": "exploration"} if command == "make explore" else {}
    if not title or len(title) > 120:
        raise ValueError("board title must contain 1-120 characters")
    if status not in BOARD_STATUSES:
        raise ValueError("board status must be todo, doing, done, or blocked")
    if priority not in BOARD_PRIORITIES:
        raise ValueError("board priority must be low, normal, or high")
    if command not in BOARD_COMMANDS:
        raise ValueError("board command is not allowlisted")
    if not isinstance(parameters, dict):
        raise ValueError("board parameters must be an object")
    allowed_parameters = BOARD_COMMANDS[command]["parameters"]
    cleaned_parameters = {}
    for key, value in parameters.items():
        if key not in allowed_parameters or not isinstance(value, str):
            raise ValueError(f"unsupported parameters for {command}")
        if len(value) > 200:
            raise ValueError("board parameter values must be under 200 characters")
        if value:
            cleaned_parameters[key] = value
    if command == "make explore":
        run_id = cleaned_parameters.get("RUN_ID", "")
        if not RUN_ID_PATTERN.fullmatch(run_id):
            raise ValueError("make explore requires an alphanumeric RUN_ID")
        probe_users = cleaned_parameters.get("SUPPORT_PROBE_USERS", "")
        if probe_users and (not re.fullmatch(r"[0-9]{1,7}", probe_users) or int(probe_users) < 1):
            raise ValueError("SUPPORT_PROBE_USERS must be a positive integer")
    if command == "create_run_record.py":
        for key in ("--variant", "--run-id"):
            if not RUN_ID_PATTERN.fullmatch(cleaned_parameters.get(key, "")):
                raise ValueError(f"create_run_record.py requires {key}")
        if cleaned_parameters.get("--status", "planned") not in {"planned", "running", "completed", "failed"}:
            raise ValueError("--status is not a valid run-record status")
    if len(owner) > 40 or len(notes) > 1000:
        raise ValueError("board owner or notes is too long")
    return {
        "title": title,
        "status": status,
        "owner": owner,
        "priority": priority,
        "notes": notes,
        "command": command,
        "parameters": cleaned_parameters,
        "last_run": (current or {}).get("last_run"),
    }

def build_board_command(item: dict) -> tuple[list[str], dict[str, str]]:
    command = item["command"]
    parameters = item["parameters"]
    spec = BOARD_COMMANDS[command]
    if "target" in spec:
        environment = os.environ.copy()
        environment.update(parameters)
        return ["make", spec["target"]], environment
    arguments = ["python3", *spec["args"]]
    for flag in ("--variant", "--run-id", "--status"):
        value = parameters.get(flag)
        if value:
            arguments.extend([flag, value])
    return arguments, os.environ.copy()


def board_run_key(item: dict) -> str | None:
    parameters = item["parameters"]
    if item["command"] == "make explore":
        return parameters.get("RUN_ID")
    if item["command"] == "create_run_record.py":
        return parameters.get("--run-id")
    return None


def serialize_job(job: dict) -> dict:
    return {key: value for key, value in job.items() if key not in {"process", "run_key"}}


def finish_job_locked(job_id: str) -> dict | None:
    job = ACTIVE_JOBS.get(job_id)
    if job is None:
        return None
    returncode = job["process"].poll()
    if returncode is None:
        return serialize_job(job)
    job["status"] = "completed" if returncode == 0 else "failed"
    job["returncode"] = returncode
    job["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    board = load_board()
    for item in board["items"]:
        last_run = item.get("last_run") or {}
        if last_run.get("job_id") == job_id:
            last_run.update(
                {
                    "status": job["status"],
                    "returncode": returncode,
                    "finished_at_utc": job["finished_at_utc"],
                }
            )
            item["last_run"] = last_run
            save_board(board)
            break
    ACTIVE_JOBS.pop(job_id, None)
    return serialize_job(job)


def reap_finished_jobs_locked() -> None:
    for job_id in list(ACTIVE_JOBS):
        if ACTIVE_JOBS[job_id]["process"].poll() is not None:
            finish_job_locked(job_id)


def board_job_status(job_id: str) -> dict:
    with BOARD_LOCK:
        job = ACTIVE_JOBS.get(job_id)
        if job is not None:
            if job["process"].poll() is not None:
                return finish_job_locked(job_id) or {}
            return serialize_job(job)
        board = load_board()
        for item in board["items"]:
            last_run = item.get("last_run") or {}
            if last_run.get("job_id") == job_id:
                return {"job_id": job_id, "item_id": item["id"], **last_run}
    raise FileNotFoundError(job_id)


def launch_board_item(item_id: str) -> tuple[dict, str]:
    with BOARD_LOCK:
        reap_finished_jobs_locked()
        board = load_board()
        current_index = next(
            (index for index, item in enumerate(board["items"]) if item.get("id") == item_id),
            None,
        )
        if current_index is None:
            raise FileNotFoundError(item_id)
        item = {"id": item_id, **clean_board_fields(board["items"][current_index], board["items"][current_index])}
        run_key = board_run_key(item)
        for active_job in ACTIVE_JOBS.values():
            if active_job["item_id"] == item_id or (run_key and active_job["run_key"] == run_key):
                raise BoardConflictError("an equivalent board command is already running")
        arguments, environment = build_board_command(item)
        log_dir = ARTIFACT_ROOT / "dashboard_jobs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{item_id}.log"
        with log_path.open("ab") as log:
            process = subprocess.Popen(
                arguments,
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        job_id = f"{item_id}-{process.pid}"
        job = {
            "job_id": job_id,
            "item_id": item_id,
            "pid": process.pid,
            "command": " ".join(arguments),
            "log_path": str(log_path.relative_to(ROOT)),
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "process": process,
            "run_key": run_key,
        }
        ACTIVE_JOBS[job_id] = job
        item["last_run"] = {
            key: value
            for key, value in serialize_job(job).items()
            if key != "item_id"
        }
        board["items"][current_index] = item
        save_board(board)
        return item, job_id




def exploration_summary(payload: dict) -> dict:
    probe = payload.get("pilot_support_probe", {})
    dimensions = payload.get("dimensions", {})
    split = payload.get("split", {})
    inputs = payload.get("inputs", {})
    return {
        "run_id": payload.get("run_id"),
        "created_at_utc": payload.get("created_at_utc"),
        "dataset_id": inputs.get("dataset_id"),
        "dataset_version": inputs.get("dataset_version"),
        "user_count": dimensions.get("user_count", 0),
        "item_count": dimensions.get("item_count", 0),
        "rating_count": dimensions.get("rating_count", 0),
        "positive_rating_count": dimensions.get("positive_rating_count", 0),
        "evaluation_eligible_user_count": split.get("evaluation_eligible_user_count", 0),
        "probe_user_count": probe.get("user_count", 0),
        "probe_eligible_item_count": probe.get("eligible_item_count", 0),
        "support_threshold": probe.get("support_threshold", 0),
        "figure_count": len(payload.get("figures", [])),
    }


def list_explorations() -> list[dict]:
    summaries = []
    for path in sorted(EXPLORATION_ROOT.glob("*.json")):
        try:
            summaries.append(exploration_summary(load_json(path)))
        except (OSError, json.JSONDecodeError, TypeError):
            continue
    return sorted(summaries, key=lambda item: item.get("created_at_utc") or "", reverse=True)


def list_runs() -> list[dict]:
    runs = []
    for path in sorted(RUN_ROOT.glob("*.json")):
        try:
            record = load_json(path)
        except (OSError, json.JSONDecodeError, TypeError):
            continue
        runs.append(
            {
                "run_id": record.get("run_id"),
                "status": record.get("status"),
                "variant_id": record.get("configuration", {}).get("variant_id"),
                "created_at_utc": record.get("created_at_utc"),
                "worktree_clean": record.get("worktree_clean"),
                "support_check_passed": record.get("support_report", {}).get("support_check_passed"),
                "record_path": f"results/run_records/{path.name}",
            }
        )
    return sorted(runs, key=lambda item: item.get("created_at_utc") or "", reverse=True)

def proposal_payload() -> dict:
    return {
        "path": "docs/research_proposal.md",
        "markdown": PROPOSAL_PATH.read_text(encoding="utf-8"),
    }




def dashboard_summary() -> dict:
    explorations = list_explorations()
    default_run_id = "raw_snapshot_preflight"
    if not any(item.get("run_id") == default_run_id for item in explorations) and explorations:
        default_run_id = explorations[0].get("run_id")
    return {
        "service": "federated-research-dashboard",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "default_run_id": default_run_id,
        "explorations": explorations,
        "runs": list_runs(),
        "board": board_snapshot(),
    }


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "FederatedResearchDashboard/1.0"

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        request = urlsplit(self.path)
        path = unquote(request.path)
        try:
            if path == "/api/health":
                self.send_json({"status": "ok", "service": self.server_version})
            elif path == "/api/summary":
                self.send_json(dashboard_summary())
            elif path == "/api/explorations":
                self.send_json(list_explorations())
            elif path.startswith("/api/explorations/"):
                run_id = path.removeprefix("/api/explorations/")
                self.send_json(self.read_exploration(run_id))
            elif path == "/api/runs":
                self.send_json(list_runs())
            elif path == "/api/board":
                self.send_json(board_snapshot())
            elif path.startswith("/api/board/jobs/"):
                job_id = path.removeprefix("/api/board/jobs/")
                if not JOB_ID_PATTERN.fullmatch(job_id):
                    raise ValueError("invalid dashboard job id")
                self.send_json(board_job_status(job_id))
            elif path == "/api/proposal":
                self.send_json(proposal_payload())
            elif path == "/docs/research_proposal.md":
                self.send_file(ROOT, "docs/research_proposal.md", content_type="text/plain; charset=utf-8")
            elif path.startswith("/artifacts/"):
                self.send_file(ARTIFACT_ROOT, path.removeprefix("/artifacts/"))
            else:
                self.send_file(WEB_ROOT, "index.html" if path == "/" else path.removeprefix("/"))
        except FileNotFoundError:
            self.send_error(404, "Not found")
        except ValueError as exc:
            self.send_error(400, str(exc))
        except Exception as exc:  # pragma: no cover - safety net for local server errors
            self.send_error(500, str(exc))

    def require_local_origin(self) -> None:
        origin = self.headers.get("Origin")
        if origin and urlsplit(origin).hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise PermissionError("board mutations require a localhost origin")

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        try:
            self.require_local_origin()
            path = urlsplit(self.path).path
            if path.startswith("/api/board/items/") and path.endswith("/run"):
                item_id = self.board_run_id_from_path()
                item, job_id = launch_board_item(item_id)
                self.send_json({"item": item, "job_id": job_id}, status=202)
                return
            if path != "/api/board/items":
                self.send_error_json(404, "Not found")
                return
            fields = clean_board_fields(self.read_json_body())
            with BOARD_LOCK:
                board = load_board()
                item = {"id": board_item_id(fields["title"], board["items"]), **fields}
                board["items"].append(item)
                save_board(board)
            self.send_json(item, status=201)
        except PermissionError as exc:
            self.send_error_json(403, str(exc))
        except BoardConflictError as exc:
            self.send_error_json(409, str(exc))
        except FileNotFoundError:
            self.send_error_json(404, "Board item not found")
        except ValueError as exc:
            self.send_error_json(400, str(exc))
        except OSError as exc:
            self.send_error_json(500, str(exc))

    def do_PATCH(self) -> None:  # noqa: N802 - stdlib handler API
        try:
            self.require_local_origin()
            item_id = self.board_item_id_from_path()
            updates = self.read_json_body()
            with BOARD_LOCK:
                board = load_board()
                current = next((item for item in board["items"] if item.get("id") == item_id), None)
                if current is None:
                    raise FileNotFoundError(item_id)
                item = {"id": item_id, **clean_board_fields(updates, current)}
                current_index = board["items"].index(current)
                board["items"][current_index] = item
                save_board(board)
            self.send_json(item)
        except PermissionError as exc:
            self.send_error_json(403, str(exc))
        except FileNotFoundError:
            self.send_error_json(404, "Board item not found")
        except ValueError as exc:
            self.send_error_json(400, str(exc))
        except OSError as exc:
            self.send_error_json(500, str(exc))

    def do_DELETE(self) -> None:  # noqa: N802 - stdlib handler API
        try:
            self.require_local_origin()
            item_id = self.board_item_id_from_path()
            with BOARD_LOCK:
                board = load_board()
                remaining = [item for item in board["items"] if item.get("id") != item_id]
                if len(remaining) == len(board["items"]):
                    raise FileNotFoundError(item_id)
                board["items"] = remaining
                save_board(board)
            self.send_json({"deleted": item_id})
        except PermissionError as exc:
            self.send_error_json(403, str(exc))
        except FileNotFoundError:
            self.send_error_json(404, "Board item not found")
        except ValueError as exc:
            self.send_error_json(400, str(exc))
        except OSError as exc:
            self.send_error_json(500, str(exc))


    def read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 64 * 1024:
            raise ValueError("request body must be a JSON object under 64 KiB")
        payload = json.loads(self.rfile.read(length))
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def board_item_id_from_path(self) -> str:
        prefix = "/api/board/items/"
        path = unquote(urlsplit(self.path).path)
        if not path.startswith(prefix):
            raise FileNotFoundError(path)
        item_id = path.removeprefix(prefix)
        if not BOARD_ID_PATTERN.fullmatch(item_id):
            raise ValueError("invalid board item id")
        return item_id

    def board_run_id_from_path(self) -> str:
        prefix = "/api/board/items/"
        suffix = "/run"
        path = unquote(urlsplit(self.path).path)
        if not path.startswith(prefix) or not path.endswith(suffix):
            raise FileNotFoundError(path)
        item_id = path[len(prefix):-len(suffix)]
        if not BOARD_ID_PATTERN.fullmatch(item_id):
            raise ValueError("invalid board item id")
        return item_id
    def read_exploration(self, run_id: str) -> dict:
        if not RUN_ID_PATTERN.fullmatch(run_id):
            raise ValueError("invalid exploration run id")
        path = EXPLORATION_ROOT / f"{run_id}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        return load_json(path)

    def send_json(self, payload: object, status: int = 200) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, status: int, message: str) -> None:
        self.send_json({"error": message}, status)

    def send_file(self, root: Path, relative: str, content_type: str | None = None) -> None:
        path = safe_child(root, relative)
        if path is None or not path.is_file():
            raise FileNotFoundError(relative)
        body = path.read_bytes()
        content_type = content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        print(f"[dashboard] {self.address_string()} - {format % args}")


class DashboardServer(ThreadingHTTPServer):
    allow_reuse_address = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    server = DashboardServer((args.host, args.port), DashboardHandler)
    print(f"dashboard listening at http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n dashboard stopped", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
