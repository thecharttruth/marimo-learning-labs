#!/usr/bin/env python3
"""Rename MoLab workspace notebooks to human titles from NOTEBOOKS.yaml."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY = REPO_ROOT / "notebooks" / "trading" / "NOTEBOOKS.yaml"
MANIFEST = REPO_ROOT / ".molab-import" / "rename-manifest.json"
SCRIPTS = REPO_ROOT / "scripts"


def _load_registry() -> list[dict]:
    projects: list[dict] = []
    current: dict | None = None
    for raw in REGISTRY.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if line.startswith("  - slug:"):
            if current:
                projects.append(current)
            current = {"slug": line.split(":", 1)[1].strip()}
            continue
        if current and line.startswith("    ") and ":" in line:
            key, value = line.strip().split(":", 1)
            current[key] = value.strip()
    if current:
        projects.append(current)
    return projects


def cmd_rename(args: argparse.Namespace) -> None:
    selected = _load_registry()
    if args.slug:
        selected = [p for p in selected if p["slug"] == args.slug]
    notebooks = []
    for p in selected:
        slug = p["slug"]
        notebooks.append(
            {
                "slug": slug,
                "title": p.get("title", slug),
                "catalog_url": p.get("molab_url", ""),
                "match_names": [slug, "notebook.py", slug.replace("-", " ")],
            }
        )
        print(f"will rename: {slug} -> {p.get('title', slug)}")

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps({"notebooks": notebooks}, indent=2) + "\n", encoding="utf-8")

    subprocess.run(["bash", str(SCRIPTS / "molab-chrome.sh")], check=False)
    proc = subprocess.run(["node", str(SCRIPTS / "molab-rename.mjs"), "--manifest", str(MANIFEST)], cwd=SCRIPTS)
    if proc.returncode != 0:
        raise SystemExit("Rename failed. Try manual rename in MoLab (see docs/GITHUB.md).")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slug")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("rename")
    p.set_defaults(func=cmd_rename)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
