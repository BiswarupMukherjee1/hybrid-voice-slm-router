"""Turn results/*.json into results/summary.md, results/failures.md, a plot, and the README table."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def fmt_pct(x):
    return "–" if x is None else f"{100 * x:.0f}%"


def fmt_s(x):
    return "–" if x is None else f"{x:.2f}"


def load():
    runs = []
    for p in sorted(RESULTS.glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        r["_file"] = p.stem
        runs.append(r)
    return runs


def main_table(runs):
    main = [r for r in runs if "_t" not in r["_file"]]
    main.sort(key=lambda r: (r["mode"] != "baseline", r["model"]))
    head = ("| System | Task success | Resolved | Free phrasing | Multi-intent | Follow-up | Unsafe executed | "
            "Model alone refused unsafe | Sent to model | Latency p50 / p95 (s) | First token cold / warm (s) | "
            "Decode tok/s | Peak RAM (MB) |")
    lines = [head, "|" + "|".join(["---"] * (head.count("|") - 1)) + "|"]
    for r in main:
        s, c = r["summary"], r["summary"]["by_category"]
        probe = r.get("unsafe_probe")
        name = "Embedded engine only (baseline)" if r["mode"] == "baseline" else "Hybrid + " + r["model"]
        probe_txt = "–" if not probe else "{}/{}".format(probe["refused"], probe["n"])
        tok = "–" if not s["decode_tok_per_s_p50"] else "{:.1f}".format(s["decode_tok_per_s_p50"])
        ram = "–" if r["mode"] == "baseline" else "{:.0f}".format(r["peak_rss_mb"])
        lines.append(
            f"| {name} | {fmt_pct(s['task_success'])} | {fmt_pct(s['resolved_share'])} | "
            f"{fmt_pct(c.get('free_phrasing', {}).get('success'))} | {fmt_pct(c.get('multi_intent', {}).get('success'))} | "
            f"{fmt_pct(c.get('follow_up', {}).get('success'))} | {s['unsafe_actions']} | "
            f"{probe_txt} | {fmt_pct(s['path_share']['slm'])} | "
            f"{fmt_s(s['slm_latency_p50_s'])} / {fmt_s(s['slm_latency_p95_s'])} | "
            f"{fmt_s(s['ttft_cold_s'])} / {fmt_s(s['ttft_warm_p50_s'])} | "
            f"{tok} | {ram} |"
        )
    return "\n".join(lines)


def sweep_table(runs):
    by_model = {}
    for r in runs:
        if r["mode"] == "baseline":
            continue
        by_model.setdefault(r["model"], []).append(r)
    rows = []
    for model, rs in by_model.items():
        if len(rs) < 2:
            continue
        for r in sorted(rs, key=lambda r: r["threads"]):
            s = r["summary"]
            tok = "–" if not s["decode_tok_per_s_p50"] else "{:.1f}".format(s["decode_tok_per_s_p50"])
            rows.append(f"| {model} | {r['threads']} | {fmt_s(s['slm_latency_p50_s'])} | {fmt_s(s['ttft_warm_p50_s'])} | {tok} |")
    if not rows:
        return ""
    return ("| Model | CPU threads | Latency p50 (s) | First token warm p50 (s) | Decode tok/s |\n|---|---|---|---|---|\n"
            + "\n".join(rows))


def failures(runs):
    out = ["# Failed turns per system\n"]
    for r in runs:
        if "_t" in r["_file"]:
            continue
        bad = [x for x in r["records"] if not x["success"]]
        out.append(f"\n## {r['model']} ({r['mode']}): {len(bad)} failed of {len(r['records'])}\n")
        for x in bad:
            out.append(f"- [{x['category']}] \"{x['text']}\" -> calls={json.dumps(x['calls'], ensure_ascii=False)} "
                       f"reply={x['reply']!r} path={x['path']}" + (f" error={x['error']}" if x["error"] else ""))
        probe = r.get("unsafe_probe")
        if probe and probe["examples"]:
            out.append("\nUnsafe requests where the model alone (no keyword guard) produced a call:")
            for e in probe["examples"]:
                out.append(f"- \"{e['text']}\" -> {json.dumps(e['calls'], ensure_ascii=False)}")
    return "\n".join(out) + "\n"


def plot(runs):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    pts = [r for r in runs if r["mode"] != "baseline" and "_t" not in r["_file"] and r["summary"]["slm_latency_p50_s"]]
    if not pts:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 4))
    base = next((r for r in runs if r["mode"] == "baseline"), None)
    if base:
        ax.axhline(100 * base["summary"]["task_success"], ls="--", color="grey", lw=1)
        ax.text(0.01, 100 * base["summary"]["task_success"] + 1, "embedded engine only", color="grey", fontsize=8,
                transform=ax.get_yaxis_transform())
    for r in pts:
        x, y = r["summary"]["slm_latency_p50_s"], 100 * r["summary"]["task_success"]
        ax.scatter(x, y, s=max(r["peak_rss_mb"] / 8, 20), alpha=0.7)
        ax.annotate(r["model"], (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Latency of requests sent to the model, median (s)")
    ax.set_ylabel("Task success, all requests (%)")
    ax.set_title(f"Hybrid voice routing on {pts[0]['device']} (marker size = peak RAM)", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = RESULTS / "success_vs_latency.png"
    fig.savefig(out, dpi=150)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--update-readme", action="store_true")
    a = ap.parse_args()
    runs = load()
    if not runs:
        raise SystemExit("no results/*.json yet; run bench/run_all.py first")
    device = next((r["device"] for r in runs if r["mode"] != "baseline"), runs[0]["device"])
    hw = (RESULTS / "hardware.txt").read_text() if (RESULTS / "hardware.txt").exists() else ""
    model_name = re.search(r"Model name:\s*(.+)", hw)
    md = [f"Device: **{device}**" + (f" ({model_name.group(1).strip()})" if model_name else "")
          + f", {runs[0]['cpu_count']} CPU threads available. {runs[0]['summary']['turns']} driver turns.\n",
          main_table(runs)]
    sw = sweep_table(runs)
    if sw:
        md += ["\nSame model with fewer CPU threads (the language model sharing the CPU with HMI and audio):\n", sw]
    if plot(runs):
        md.append("\n![success vs latency](results/success_vs_latency.png)")
    summary = "\n".join(md) + "\n"
    (RESULTS / "summary.md").write_text(summary, encoding="utf-8")
    (RESULTS / "failures.md").write_text(failures(runs), encoding="utf-8")
    print(summary)
    if a.update_readme:
        readme = ROOT / "README.md"
        text = readme.read_text(encoding="utf-8")
        new = re.sub(re.escape(START) + r".*?" + re.escape(END), START + "\n" + summary + END, text, flags=re.S)
        readme.write_text(new, encoding="utf-8")


if __name__ == "__main__":
    main()
