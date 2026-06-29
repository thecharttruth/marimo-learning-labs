#!/usr/bin/env python3
"""Mirror notebooks from GitHub into MoLab workspace (synced notebooks).

After setup, push to GitHub and MoLab reflects changes automatically.

Usage:
  ./scripts/molab-github-mirror.sh login
  ./scripts/molab-github-mirror.sh mirror
  ./scripts/molab-github-mirror.sh mirror --slug btc-adaptive-supertrend-lab
  ./scripts/molab-github-mirror.sh status

Registry: notebooks/trading/NOTEBOOKS.yaml
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = REPO_ROOT / "notebooks" / "trading" / "NOTEBOOKS.yaml"
IMPORT_DIR = REPO_ROOT / ".molab-import"
MANIFEST = IMPORT_DIR / "mirror-manifest.json"
RESULTS = IMPORT_DIR / "mirror-results.json"
SCRIPTS = REPO_ROOT / "scripts"
NODE_MIRROR = SCRIPTS / "molab-github-mirror.mjs"
GITHUB_REPO = "thecharttruth/marimo-learning-labs"
GITHUB_BRANCH = "main"


def _load_registry(path: Path) -> list[dict]:
    projects: list[dict] = []
    current: dict | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
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
        value = value.strip().strip('"')
        current[key] = value
    if current:
        projects.append(current)
    return projects


def github_blob_url(project: dict) -> str:
    """Use slug.py on GitHub so MoLab does not label every mirror 'notebook.py'."""
    local_path = project["local_path"].rstrip("/")
    slug = project["slug"]
    return (
        f"https://github.com/{GITHUB_REPO}/blob/{GITHUB_BRANCH}/"
        f"{local_path}/{slug}.py"
    )


def github_preview_url(project: dict) -> str:
    local_path = project["local_path"].rstrip("/")
    slug = project["slug"]
    return (
        f"https://molab.marimo.io/github/{GITHUB_REPO}/blob/{GITHUB_BRANCH}/"
        f"{local_path}/{slug}.py"
    )


def _is_github_sync(project: dict) -> bool:
    return project.get("cloud_kind", "") == "github_sync"


def cmd_status(args: argparse.Namespace) -> None:
    registry = Path(args.registry)
    print(f"{'slug':<34} {'sync':<12} molab_url")
    for p in _load_registry(registry):
        kind = "github_sync" if _is_github_sync(p) else p.get("cloud_kind", "upload")
        url = p.get("molab_url", "-")[:52]
        print(f"{p['slug']:<34} {kind:<12} {url}")


def cmd_stage(args: argparse.Namespace) -> None:
    registry = Path(args.registry)
    projects = _load_registry(registry)
    selected = [p for p in projects if not args.slug or p["slug"] == args.slug]
    if args.slug and not selected:
        raise SystemExit(f"Unknown slug '{args.slug}' in {registry}")

    if not args.force:
        already = [p["slug"] for p in selected if _is_github_sync(p)]
        if already and not args.slug:
            print(f"Already github_sync (use --force to remirror): {', '.join(already)}")
            selected = [p for p in selected if not _is_github_sync(p)]
        elif already and args.slug:
            print(f"{args.slug} already github_sync. Use --force to create a new mirror.")
            return

    if not selected:
        print("Nothing to mirror.")
        return

    IMPORT_DIR.mkdir(parents=True, exist_ok=True)
    notebooks = []
    for project in selected:
        blob = github_blob_url(project)
        notebooks.append(
            {
                "slug": project["slug"],
                "title": project.get("title", project["slug"]),
                "github_url": blob,
                "github_preview_url": github_preview_url(project),
                "previous_molab_url": project.get("molab_url", ""),
            }
        )
        print(f"staged mirror {project['slug']}")
        print(f"  {blob}")

    MANIFEST.write_text(
        json.dumps({"registry": str(registry), "notebooks": notebooks}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"manifest -> {MANIFEST}")


def _ensure_chrome() -> None:
    subprocess.run(["bash", str(SCRIPTS / "molab-chrome.sh")], check=False)


def cmd_login(_: argparse.Namespace) -> None:
    _ensure_chrome()
    proc = subprocess.run(
        ["node", str(NODE_MIRROR), "--login-only"],
        cwd=SCRIPTS,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit("Login failed. Run: ./scripts/molab-github-mirror.sh login")


def cmd_mirror(args: argparse.Namespace) -> None:
    cmd_stage(args)
    if not MANIFEST.exists():
        return
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if not payload.get("notebooks"):
        return

    _ensure_chrome()
    proc = subprocess.run(
        ["node", str(NODE_MIRROR), "--manifest", str(MANIFEST), "--results", str(RESULTS)],
        cwd=SCRIPTS,
    )
    if proc.returncode != 0:
        raise SystemExit(
            "Mirror failed. Ensure Chrome is logged in:\n"
            "  ./scripts/molab-github-mirror.sh login\n"
            "Then retry:\n"
            "  ./scripts/molab-github-mirror.sh mirror"
        )

    if not RESULTS.exists():
        raise SystemExit(f"Missing results: {RESULTS}")

    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    registry_path = Path(args.registry)
    for row in results.get("notebooks", []):
        row["github_preview_url"] = row.get("github_preview_url") or github_preview_url(
            next(p for p in _load_registry(registry_path) if p["slug"] == row["slug"])
        )
        _patch_registry(registry_path, row)
        _patch_metadata(row)
        print(f"updated registry + metadata for {row['slug']}")

    print("\nRenaming MoLab cards to human titles...")
    rename_proc = subprocess.run(
        [sys.executable, str(SCRIPTS / "molab-rename.py"), "rename"],
        cwd=REPO_ROOT,
        check=False,
    )
    if rename_proc.returncode != 0:
        print("Rename step failed — set titles manually in MoLab (see docs/GITHUB.md).")


def _patch_registry(registry_path: Path, row: dict) -> None:
    slug = row["slug"]
    catalog_url = row["catalog_url"]
    github_url = row["github_url"]
    preview_url = row.get("github_preview_url", "")

    text = registry_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    out: list[str] = []
    in_slug = False
    for line in lines:
        if line.strip() == f"- slug: {slug}":
            in_slug = True
            out.append(line)
            continue
        if in_slug and line.startswith("  - slug:"):
            in_slug = False
        if in_slug and re.match(r"^\s+molab_url:", line):
            out.append(f"    molab_url: {catalog_url}")
            continue
        if in_slug and re.match(r"^\s+github_url:", line):
            out.append(f"    github_url: {github_url}")
            continue
        if in_slug and re.match(r"^\s+github_preview_url:", line):
            if preview_url:
                out.append(f"    github_preview_url: {preview_url}")
            continue
        if in_slug and re.match(r"^\s+previous_molab_url:", line):
            if row.get("previous_molab_url"):
                out.append(f"    previous_molab_url: {row['previous_molab_url']}")
            continue
        if in_slug and re.match(r"^\s+cloud_kind:", line):
            out.append("    cloud_kind: github_sync")
            continue
        out.append(line)

    registry_path.write_text("\n".join(out) + ("\n" if text.endswith("\n") else ""), encoding="utf-8")


def _patch_metadata(row: dict) -> None:
    registry = _load_registry(DEFAULT_REGISTRY)
    project = next((p for p in registry if p["slug"] == row["slug"]), None)
    if not project:
        return
    meta_path = REPO_ROOT / project["local_path"] / "metadata.yaml"
    if not meta_path.exists():
        return

    catalog_url = row["catalog_url"]
    text = meta_path.read_text(encoding="utf-8")
    text = re.sub(
        r"^(\s+molab_url:\s*).*$",
        rf"\1{catalog_url}",
        text,
        flags=re.MULTILINE,
    )
    if re.search(r"^\s+cloud_kind:", text, flags=re.MULTILINE):
        text = re.sub(r"^(\s+cloud_kind:\s*).*$", r"\1github_sync", text, flags=re.MULTILINE)
    if "notebook_url:" in text:
        text = re.sub(
            r"^(\s+notebook_url:\s*).*$",
            rf"\1{row['github_url']}",
            text,
            flags=re.MULTILINE,
        )

    meta_path.write_text(text, encoding="utf-8")

    sync_state = REPO_ROOT / project["local_path"] / ".sync" / "state.json"
    sync_state.parent.mkdir(parents=True, exist_ok=True)
    state = {}
    if sync_state.exists():
        state = json.loads(sync_state.read_text(encoding="utf-8"))
    state.update(
        {
            "molab_url": catalog_url,
            "cloud_kind": "github_sync",
            "github_url": row["github_url"],
            "mirrored_from_github_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    sync_state.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mirror GitHub notebooks into MoLab.")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--slug", help="Only mirror this slug")
    parser.add_argument("--force", action="store_true", help="Create new mirror even if github_sync")
    sub = parser.add_subparsers(dest="command", required=True)

    for name, fn, help_text in [
        ("status", cmd_status, "Show github_sync vs catalog per slug"),
        ("stage", cmd_stage, "Build mirror manifest from registry"),
        ("login", cmd_login, "Confirm MoLab login in Chrome"),
        ("mirror", cmd_mirror, "Stage + mirror from GitHub + update registry"),
    ]:
        p = sub.add_parser(name, help=help_text)
        p.set_defaults(func=fn)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
