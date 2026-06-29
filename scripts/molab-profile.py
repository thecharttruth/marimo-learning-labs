#!/usr/bin/env python3
"""Publish local marimo notebooks to MoLab profile (catalog nb_* URLs).

MoLab has no public API for profile creation. This tool stages notebooks from a
registry YAML, drives the MoLab Import UI via Playwright + Chrome CDP, then
updates registry metadata with the new catalog URLs.

Usage:
  ./scripts/molab-profile.sh login
  ./scripts/molab-profile.sh publish
  ./scripts/molab-profile.sh publish --slug btc-adaptive-supertrend-lab
  ./scripts/molab-profile.sh status

Registry: notebooks/trading/NOTEBOOKS.yaml (override with --registry)
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = REPO_ROOT / "notebooks" / "trading" / "NOTEBOOKS.yaml"
IMPORT_DIR = REPO_ROOT / ".molab-import"
MANIFEST = IMPORT_DIR / "manifest.json"
RESULTS = IMPORT_DIR / "results.json"
SCRIPTS = REPO_ROOT / "scripts"
MARIMO = REPO_ROOT / "notebooks/trading/global-macro-opportunity-set-lab/.venv/bin/marimo"
NODE_IMPORT = SCRIPTS / "molab-profile-import.mjs"


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


def _is_catalog_url(url: str) -> bool:
    return "molab.marimo.io/notebooks/nb_" in url


def _needs_publish(project: dict, force: bool) -> bool:
    if force:
        return True
    url = project.get("molab_url", "")
    kind = project.get("cloud_kind", "")
    if _is_catalog_url(url):
        return False
    return kind in {"", "sandbox", "unpublished", "ephemeral"} or "sb.molab.run" in url


def _run_marimo_check(notebook: Path) -> None:
    if not MARIMO.exists():
        print(f"skip marimo check (missing {MARIMO})")
        return
    proc = subprocess.run(
        [str(MARIMO), "check", "--strict", str(notebook)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(f"marimo check failed for {notebook}:\n{proc.stdout}\n{proc.stderr}")


def cmd_status(args: argparse.Namespace) -> None:
    registry = Path(args.registry)
    rows = []
    for p in _load_registry(registry):
        url = p.get("molab_url", "-")
        state = "catalog" if _is_catalog_url(url) else "needs_publish"
        rows.append((p["slug"], state, url[:56]))
    print(f"{'slug':<34} {'profile':<14} molab_url")
    for slug, state, url in rows:
        print(f"{slug:<34} {state:<14} {url}")


def cmd_stage(args: argparse.Namespace) -> None:
    registry = Path(args.registry)
    projects = _load_registry(registry)
    selected = [p for p in projects if not args.slug or p["slug"] == args.slug]
    if args.slug and not selected:
        raise SystemExit(f"Unknown slug '{args.slug}' in {registry}")

    to_publish = [p for p in selected if _needs_publish(p, args.force)]
    if not to_publish:
        print("All selected notebooks already have catalog URLs (use --force to republish).")
        return

    IMPORT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_notebooks = []
    for project in to_publish:
        slug = project["slug"]
        src = REPO_ROOT / project["local_path"] / "notebook.py"
        if not src.exists():
            raise SystemExit(f"Missing notebook: {src}")
        _run_marimo_check(src)
        dst = IMPORT_DIR / f"{slug}.py"
        shutil.copy2(src, dst)
        manifest_notebooks.append(
            {
                "slug": slug,
                "title": project.get("title", slug),
                "local_path": project["local_path"],
                "staging_path": str(dst),
                "existing_url": project.get("molab_url", ""),
            }
        )
        print(f"staged {slug}")

    MANIFEST.write_text(
        json.dumps({"registry": str(registry), "notebooks": manifest_notebooks}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"manifest -> {MANIFEST}")


def _ensure_chrome() -> None:
    chrome_sh = SCRIPTS / "molab-chrome.sh"
    subprocess.run(["bash", str(chrome_sh)], check=False)


def cmd_login(_: argparse.Namespace) -> None:
    _ensure_chrome()
    proc = subprocess.run(["node", str(NODE_IMPORT), "--login-only"], cwd=SCRIPTS, check=False)
    if proc.returncode != 0:
        raise SystemExit(
            "Login not confirmed. Complete Google sign-in in the Chrome window, then rerun:\n"
            "  ./scripts/molab-profile.sh login"
        )


def cmd_publish(args: argparse.Namespace) -> None:
    cmd_stage(args)
    if not MANIFEST.exists():
        return

    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if not payload.get("notebooks"):
        return

    _ensure_chrome()
    import_cmd = ["node", str(NODE_IMPORT), "--manifest", str(MANIFEST), "--results", str(RESULTS)]
    if args.debug:
        import_cmd.append("--debug")
    proc = subprocess.run(import_cmd, cwd=SCRIPTS)
    if proc.returncode != 0:
        raise SystemExit(
            "Publish failed. Ensure you are logged in:\n"
            "  ./scripts/molab-profile.sh login\n"
            "Then retry:\n"
            "  ./scripts/molab-profile.sh publish"
        )

    if not RESULTS.exists():
        raise SystemExit(f"Missing results file: {RESULTS}")

    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    registry_path = Path(args.registry)
    for row in results.get("notebooks", []):
        _patch_registry(registry_path, row["slug"], row["catalog_url"])
        _patch_metadata(row["slug"], row["catalog_url"])
        print(f"updated registry + metadata for {row['slug']}")


def _patch_registry(registry_path: Path, slug: str, catalog_url: str) -> None:
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
        if in_slug and re.match(r"^\s+cloud_kind:", line):
            out.append("    cloud_kind: catalog")
            continue
        out.append(line)
    registry_path.write_text("\n".join(out) + ("\n" if text.endswith("\n") else ""), encoding="utf-8")


def _patch_metadata(slug: str, catalog_url: str) -> None:
    registry = _load_registry(DEFAULT_REGISTRY)
    project = next((p for p in registry if p["slug"] == slug), None)
    if not project:
        return
    meta_path = REPO_ROOT / project["local_path"] / "metadata.yaml"
    if not meta_path.exists():
        return
    text = meta_path.read_text(encoding="utf-8")
    text = re.sub(
        r"^(\s+molab_notebook_url:\s*).*$",
        rf"\1{catalog_url}",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"^(\s+molab_url:\s*).*$",
        rf"\1{catalog_url}",
        text,
        flags=re.MULTILINE,
    )
    if "cloud_kind:" in text:
        text = re.sub(r"^(\s+cloud_kind:\s*).*$", r"\1catalog", text, flags=re.MULTILINE)
    meta_path.write_text(text, encoding="utf-8")

    sync_state = REPO_ROOT / project["local_path"] / ".sync" / "state.json"
    sync_state.parent.mkdir(parents=True, exist_ok=True)
    state = {}
    if sync_state.exists():
        state = json.loads(sync_state.read_text(encoding="utf-8"))
    state.update(
        {
            "molab_url": catalog_url,
            "cloud_kind": "catalog",
            "published_to_profile_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    sync_state.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish local notebooks to MoLab profile.")
    parser.add_argument(
        "--registry",
        default=str(DEFAULT_REGISTRY),
        help="NOTEBOOKS.yaml registry path",
    )
    parser.add_argument("--slug", help="Only publish this slug")
    parser.add_argument("--force", action="store_true", help="Republish even if catalog URL exists")
    parser.add_argument("--debug", action="store_true", help="Dump MoLab UI debug info during import")
    sub = parser.add_subparsers(dest="command", required=True)

    for name, fn, help_text in [
        ("status", cmd_status, "Show catalog vs needs_publish per registry slug"),
        ("stage", cmd_stage, "Validate and stage notebooks into .molab-import/"),
        ("login", cmd_login, "Open Chrome for MoLab login (one-time)"),
        ("publish", cmd_publish, "Stage + import to MoLab profile + update registry"),
    ]:
        p = sub.add_parser(name, help=help_text)
        p.set_defaults(func=fn)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
