#!/usr/bin/env python3
"""Sync marimo notebook.py between local project folders and live MoLab sessions.

Usage:
  MARIMO_TOKEN=... python3 scripts/molab-sync.py status
  MARIMO_TOKEN=... python3 scripts/molab-sync.py pull --slug global-macro-opportunity-set-lab
  MARIMO_TOKEN=... python3 scripts/molab-sync.py push --slug global-macro-opportunity-set-lab
  MARIMO_TOKEN=... python3 scripts/molab-sync.py sync --slug global-macro-opportunity-set-lab

Never commit tokens. MoLab sandbox URLs (sb-*.sb.molab.run) are execution surfaces;
this repo is the durable backup. Pull after cloud edits. Push before sharing a run.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY = REPO_ROOT / "notebooks" / "trading" / "NOTEBOOKS.yaml"
EXECUTE_CODE = REPO_ROOT / ".agents" / "skills" / "marimo-pair" / "scripts" / "execute-code.sh"


def _load_registry_text() -> str:
    return REGISTRY.read_text(encoding="utf-8")


def _parse_registry() -> list[dict]:
    """Minimal YAML parser for our NOTEBOOKS.yaml shape (no PyYAML dependency)."""
    projects: list[dict] = []
    current: dict | None = None
    for raw in _load_registry_text().splitlines():
        line = raw.rstrip()
        if line.startswith("  - slug:"):
            if current:
                projects.append(current)
            current = {"slug": line.split(":", 1)[1].strip()}
            continue
        if current is None or not line.startswith("    "):
            continue
        if ":" not in line:
            continue
        key, value = line.strip().split(":", 1)
        value = value.strip()
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        current[key] = value
    if current:
        projects.append(current)
    return projects


def _project_for_slug(slug: str) -> dict:
    for project in _parse_registry():
        if project.get("slug") == slug:
            return project
    raise SystemExit(f"Unknown slug '{slug}'. See {REGISTRY}")


def _local_notebook(project: dict) -> Path:
    return REPO_ROOT / project["local_path"] / "notebook.py"


def _sync_state_path(project: dict) -> Path:
    return REPO_ROOT / project["local_path"] / ".sync" / "state.json"


def _execution_url(project: dict, override: str | None) -> str:
    if override:
        return override.rstrip("/")
    state_path = _sync_state_path(project)
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        cached = state.get("execution_url", "")
        if cached:
            return cached.rstrip("/")
    url = project.get("molab_url", "").rstrip("/")
    if "sb.molab.run" in url:
        return url
    raise SystemExit(
        f"Catalog URL {url} is not a live marimo server. "
        "Open the notebook in MoLab, copy the sb-*.sb.molab.run URL from the browser, "
        "and pass --url <sandbox_url>."
    )


def _cell_code_hash(text: str) -> str:
    """Hash notebook logic using @app.cell bodies (ignores MoLab metadata drift)."""
    parts = re.findall(
        r"@app\.cell(?:\([^)]*\))?\s*\ndef [^\n]+:\s*\n(.*?)(?=\n@app\.cell|\nif __name__|\Z)",
        text,
        flags=re.DOTALL,
    )
    joined = "\n---CELL---\n".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _http_get(url: str, token: str, path: str) -> tuple[int, bytes]:
    req = urllib.request.Request(
        f"{url.rstrip('/')}{path}",
        headers={
            "Authorization": f"Bearer {token}",
            "User-Agent": "marimo-molab-sync/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _list_sessions(url: str, token: str) -> dict:
    status, body = _http_get(url, token, "/api/sessions")
    if status != 200:
        raise SystemExit(f"Failed to list sessions at {url} (HTTP {status}). Is the notebook open in MoLab?")
    return json.loads(body.decode("utf-8"))


def _session_id(url: str, token: str, explicit: str | None) -> str:
    if explicit:
        return explicit
    sessions = _list_sessions(url, token)
    if not sessions:
        raise SystemExit(f"No active session at {url}. Open the notebook in MoLab first.")
    if len(sessions) > 1:
        keys = ", ".join(sessions)
        raise SystemExit(f"Multiple sessions at {url}: {keys}. Pass --session explicitly.")
    return next(iter(sessions))


def _run_execute(url: str, token: str, session_id: str, code: str) -> str:
    if not EXECUTE_CODE.exists():
        raise SystemExit(f"Missing execute-code helper: {EXECUTE_CODE}")
    env = {**dict(**__import__("os").environ), "MARIMO_TOKEN": token}
    proc = subprocess.run(
        ["bash", str(EXECUTE_CODE), "--url", url, "--session", session_id],
        input=code,
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(proc.stderr.strip() or proc.stdout.strip() or "execute-code failed")
    return proc.stdout


def pull_cloud_text(url: str, token: str, session_id: str | None) -> str:
    sid = _session_id(url, token, session_id)
    status, body = _http_get(url, token, "/marimo/notebook.py")
    if status == 200 and b"import marimo" in body:
        return body.decode("utf-8")
    return _run_execute(
        url,
        token,
        sid,
        "from pathlib import Path\nprint(Path('/marimo/notebook.py').read_text(), end='')",
    )


def push_cloud_text(url: str, token: str, session_id: str | None, text: str) -> None:
    sid = _session_id(url, token, session_id)
    encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
    code = (
        "import base64\n"
        "from pathlib import Path\n"
        f'Path("/marimo/notebook.py").write_bytes(base64.b64decode("{encoded}"))\n'
        'print("cloud notebook.py bytes:", Path("/marimo/notebook.py").stat().st_size)\n'
    )
    out = _run_execute(url, token, sid, code)
    if "cloud notebook.py bytes:" not in out:
        raise SystemExit(f"Unexpected push response: {out[:300]}")


def _write_sync_state(project: dict, **fields) -> None:
    path = _sync_state_path(project)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {}
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
    state.update(fields)
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def cmd_status(args: argparse.Namespace) -> None:
    token = __import__("os").environ.get("MARIMO_TOKEN", "")
    rows = []
    for project in _parse_registry():
        local_path = _local_notebook(project)
        local_exists = local_path.exists()
        local_hash = _cell_code_hash(local_path.read_text(encoding="utf-8")) if local_exists else None
        cloud_hash = None
        cloud_ok = "skipped"
        if token and not args.offline:
            try:
                exec_url = _execution_url(project, getattr(args, "url", None))
                cloud_text = pull_cloud_text(exec_url, token, args.session)
                cloud_hash = _cell_code_hash(cloud_text)
                cloud_ok = "match" if local_hash == cloud_hash else "drift"
            except SystemExit as exc:
                cloud_ok = f"error: {exc}"
        rows.append((project["slug"], local_exists, cloud_ok, local_hash, cloud_hash))
    print(f"{'slug':<34} {'local':<6} {'cloud':<10} local_hash   cloud_hash")
    for slug, local_exists, cloud_ok, lh, ch in rows:
        print(
            f"{slug:<34} {str(local_exists):<6} {cloud_ok:<10} "
            f"{(lh or '-')[:12]} {(ch or '-')[:12]}"
        )


def cmd_pull(args: argparse.Namespace) -> None:
    token = _require_token()
    project = _project_for_slug(args.slug)
    local_path = _local_notebook(project)
    exec_url = _execution_url(project, args.url)
    text = pull_cloud_text(exec_url, token, args.session)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_text(text, encoding="utf-8")
    _write_sync_state(
        project,
        last_pull_at=datetime.now(timezone.utc).isoformat(),
        local_cell_hash=_cell_code_hash(text),
        cloud_cell_hash=_cell_code_hash(text),
        molab_url=project["molab_url"],
        execution_url=exec_url,
    )
    print(f"Pulled -> {local_path} ({len(text.splitlines())} lines)")


def cmd_push(args: argparse.Namespace) -> None:
    token = _require_token()
    project = _project_for_slug(args.slug)
    local_path = _local_notebook(project)
    if not local_path.exists():
        raise SystemExit(f"Missing local notebook: {local_path}")
    text = local_path.read_text(encoding="utf-8")
    exec_url = _execution_url(project, args.url)
    push_cloud_text(exec_url, token, args.session, text)
    _write_sync_state(
        project,
        last_push_at=datetime.now(timezone.utc).isoformat(),
        local_cell_hash=_cell_code_hash(text),
        cloud_cell_hash=_cell_code_hash(text),
        molab_url=project["molab_url"],
        execution_url=exec_url,
    )
    print(f"Pushed <- {local_path} via {exec_url}")
    print("Reload the MoLab tab if the UI does not pick up file changes immediately.")


def cmd_sync(args: argparse.Namespace) -> None:
    """Push local to cloud, then pull back to confirm round-trip hash."""
    token = _require_token()
    project = _project_for_slug(args.slug)
    local_path = _local_notebook(project)
    text = local_path.read_text(encoding="utf-8")
    local_hash = _cell_code_hash(text)
    exec_url = _execution_url(project, args.url)
    push_cloud_text(exec_url, token, args.session, text)
    pulled = pull_cloud_text(exec_url, token, args.session)
    local_path.write_text(pulled, encoding="utf-8")
    cloud_hash = _cell_code_hash(pulled)
    _write_sync_state(
        project,
        last_sync_at=datetime.now(timezone.utc).isoformat(),
        local_cell_hash=local_hash,
        cloud_cell_hash=cloud_hash,
        molab_url=project["molab_url"],
        execution_url=exec_url,
    )
    if local_hash != cloud_hash:
        raise SystemExit(
            "Sync round-trip hash mismatch. MoLab may have normalized the file; "
            "compare cell bodies manually."
        )
    print(f"Synced {args.slug}: local and cloud cell hashes match ({local_hash[:12]}...)")


def _require_token() -> str:
    token = __import__("os").environ.get("MARIMO_TOKEN", "")
    if not token:
        raise SystemExit("Set MARIMO_TOKEN in the environment.")
    return token


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync local marimo notebooks with MoLab sessions.")
    parser.add_argument("--session", help="Explicit marimo session id (from /api/sessions)")
    parser.add_argument(
        "--url",
        help="Live marimo execution URL (sb-*.sb.molab.run). Required for catalog nb_* URLs.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_status = sub.add_parser("status", help="Show local vs cloud drift for all projects")
    p_status.add_argument("--offline", action="store_true", help="Only inspect local hashes")
    p_status.set_defaults(func=cmd_status)

    for name, help_text in [
        ("pull", "Pull cloud notebook.py into the local project folder"),
        ("push", "Push local notebook.py to the open MoLab session"),
        ("sync", "Push local, pull back, verify cell-body hash"),
    ]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--slug", required=True)
        p.set_defaults(func={"pull": cmd_pull, "push": cmd_push, "sync": cmd_sync}[name])

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
