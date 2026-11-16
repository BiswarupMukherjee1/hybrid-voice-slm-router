"""Download the GGUF files listed in models.yaml into models/."""

from __future__ import annotations

import argparse
import fnmatch
import sys
from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download, list_repo_files

ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    return yaml.safe_load((ROOT / "models.yaml").read_text())


def resolve_file(repo: str, pattern: str) -> str:
    files = [f for f in list_repo_files(repo) if fnmatch.fnmatch(f.lower(), pattern.lower())]
    if not files:
        raise FileNotFoundError(f"no file matching {pattern} in {repo}")
    return sorted(files, key=len)[0]


def download(selected: list[str] | None = None) -> dict[str, Path]:
    paths = {}
    for m in load_config()["models"]:
        if selected and m["name"] not in selected:
            continue
        try:
            fname = resolve_file(m["repo"], m["pattern"])
            p = hf_hub_download(m["repo"], fname, local_dir=ROOT / "models")
            paths[m["name"]] = Path(p)
            print(f"ok   {m['name']}: {m['repo']}/{fname}")
        except Exception as e:  # gated or missing repos should not stop the whole run
            if m.get("optional"):
                print(f"skip {m['name']}: {e}", file=sys.stderr)
            else:
                raise
    return paths


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="", help="comma-separated names (default: all)")
    a = ap.parse_args()
    download([s for s in a.models.split(",") if s] or None)
