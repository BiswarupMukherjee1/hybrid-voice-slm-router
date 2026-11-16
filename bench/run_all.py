"""Download models and run the full benchmark, one process per configuration.

  python bench/run_all.py                                  # everything in models.yaml
  python bench/run_all.py --models qwen2.5-0.5b-q4 --limit 20   # quick check
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.download_models import download, load_config  # noqa: E402

RESULTS = ROOT / "results"


def sh(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def hardware_info(label: str) -> None:
    RESULTS.mkdir(exist_ok=True)
    lines = [f"device: {label}", f"machine: {platform.machine()}", f"cpu_count: {os.cpu_count()}"]
    try:
        lines.append(subprocess.run(["lscpu"], capture_output=True, text=True).stdout)
    except FileNotFoundError:
        pass
    (RESULTS / "hardware.txt").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="", help="comma-separated model names (default: all)")
    ap.add_argument("--limit", type=int, help="only the first N cases (quick check)")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--device-label", default=platform.machine())
    ap.add_argument("--no-sweep", action="store_true")
    a = ap.parse_args()

    selected = [s for s in a.models.split(",") if s] or None
    paths = download(selected)
    hardware_info(a.device_label)
    common = ["--device-label", a.device_label] + (["--limit", str(a.limit)] if a.limit else [])
    py = sys.executable
    bench = str(ROOT / "bench" / "run_benchmark.py")

    sh([py, bench, "--mode", "baseline", "--out", str(RESULTS / "baseline.json"), *common])
    for name, path in paths.items():
        sh([py, bench, "--mode", "hybrid", "--model-path", str(path), "--model-name", name,
            "--threads", str(a.threads), "--unsafe-probe", "--out", str(RESULTS / f"{name}.json"), *common])

    sweep = load_config().get("thread_sweep")
    if sweep and not a.no_sweep and sweep["model"] in paths:
        for t in sweep["threads"]:
            if t < a.threads:
                sh([py, bench, "--mode", "hybrid", "--model-path", str(paths[sweep["model"]]),
                    "--model-name", sweep["model"], "--threads", str(t),
                    "--out", str(RESULTS / f"{sweep['model']}_t{t}.json"), *common])

    sh([py, str(ROOT / "bench" / "report.py"), "--update-readme"])


if __name__ == "__main__":
    main()
