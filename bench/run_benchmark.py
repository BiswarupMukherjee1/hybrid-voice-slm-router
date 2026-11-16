"""Run one system configuration over the request set and write a JSON result file.

  python bench/run_benchmark.py --mode baseline --out results/baseline.json
  python bench/run_benchmark.py --mode hybrid --model-path models/x.gguf --model-name x --threads 4 --out results/x.json

Each model runs in its own process (run_all.py does this), so the peak memory reported
here belongs to that model only.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from voice_router.router import Dialogue  # noqa: E402
from voice_router.scoring import score_turn  # noqa: E402

DATA = ROOT / "data" / "requests.jsonl"


def load_cases(limit: int | None = None) -> list[dict]:
    cases = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines() if line.strip()]
    return cases[:limit] if limit else cases


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux reports KB


def pct(values, q):
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def run(mode: str, slm, cases: list[dict], use_guard: bool = True) -> list[dict]:
    records = []
    for case in cases:
        d = Dialogue(mode, slm=slm, use_guard=use_guard)
        for i, turn in enumerate(case["turns"]):
            r = d.handle(turn["text"])
            s = score_turn(turn["category"], turn["expected"], r)
            records.append({
                "case": case["id"], "turn": i, "category": turn["category"], "text": turn["text"],
                "expected": turn["expected"], "calls": r.calls, "reply": r.reply,
                "rejected": [[c, why] for c, why in r.rejected], "error": r.error, "raw": r.raw,
                "latency_s": r.latency_s, "ttft_s": r.ttft_s, "decode_tok_per_s": r.decode_tok_per_s,
                "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens, **s,
            })
    return records


def summarize(records: list[dict]) -> dict:
    by_cat = defaultdict(list)
    for r in records:
        by_cat[r["category"]].append(r)
    slm = [r for r in records if r["path"] == "slm"]
    ttfts = [r["ttft_s"] for r in slm if r["ttft_s"] is not None]
    return {
        "turns": len(records),
        "task_success": sum(r["success"] for r in records) / len(records),
        "resolved_share": sum(r["resolved"] for r in records) / len(records),
        "unsafe_actions": sum(r["unsafe_action"] for r in records),
        "by_category": {
            cat: {"n": len(rs), "success": sum(r["success"] for r in rs) / len(rs),
                  "resolved": sum(r["resolved"] for r in rs) / len(rs)}
            for cat, rs in sorted(by_cat.items())
        },
        "path_share": {p: sum(r["path"] == p for r in records) / len(records) for p in ("guard", "baseline", "slm", "none")},
        "invalid_outputs": sum(1 for r in slm if r["error"]),
        "rejected_calls": sum(len(r["rejected"]) for r in records),
        "slm_latency_p50_s": pct([r["latency_s"] for r in slm], 0.5),
        "slm_latency_p95_s": pct([r["latency_s"] for r in slm], 0.95),
        "ttft_cold_s": ttfts[0] if ttfts else None,
        "ttft_warm_p50_s": pct(ttfts[1:], 0.5),
        "decode_tok_per_s_p50": pct([r["decode_tok_per_s"] for r in slm if r["decode_tok_per_s"]], 0.5),
        "prompt_tokens_p50": pct([r["prompt_tokens"] for r in slm if r["prompt_tokens"]], 0.5),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "hybrid", "slm"], required=True)
    ap.add_argument("--model-path")
    ap.add_argument("--model-name", default="baseline")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--limit", type=int)
    ap.add_argument("--device-label", default=platform.machine())
    ap.add_argument("--unsafe-probe", action="store_true",
                    help="also run unsafe requests with the keyword guard switched off, to see what the model alone does")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cases = load_cases(args.limit)
    slm, load_s, rss_before = None, 0.0, peak_rss_mb()
    if args.mode != "baseline":
        from voice_router.slm import LlamaCppSLM
        t0 = time.perf_counter()
        slm = LlamaCppSLM(args.model_path, n_threads=args.threads)
        load_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    records = run(args.mode, slm, cases)
    wall = time.perf_counter() - t0
    summary = summarize(records)

    probe = None
    if args.unsafe_probe and slm is not None:
        unsafe_cases = [c for c in cases if all(t["category"] == "unsafe" for t in c["turns"])]
        probe_records = run("slm", slm, unsafe_cases, use_guard=False)
        probe = {
            "n": len(probe_records),
            "refused": sum(not r["calls"] for r in probe_records),
            "unsafe_actions": sum(r["unsafe_action"] for r in probe_records),
            "examples": [{"text": r["text"], "calls": r["calls"], "reply": r["reply"]} for r in probe_records if r["calls"]][:5],
        }

    out = {
        "model": args.model_name, "mode": args.mode, "threads": args.threads, "device": args.device_label,
        "machine": platform.machine(), "cpu_count": os.cpu_count(), "python": platform.python_version(),
        "model_file_mb": (Path(args.model_path).stat().st_size / 2**20) if args.model_path else None,
        "load_s": load_s, "wall_s": wall, "peak_rss_mb": peak_rss_mb(), "rss_before_load_mb": rss_before,
        "summary": summary, "unsafe_probe": probe, "records": records,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    s = summary
    print(f"[{args.model_name} {args.mode} t={args.threads}] success {s['task_success']:.1%}  resolved {s['resolved_share']:.1%}  "
          f"unsafe actions {s['unsafe_actions']}  slm p50 {s['slm_latency_p50_s']}  wall {wall:.0f}s")


if __name__ == "__main__":
    main()
