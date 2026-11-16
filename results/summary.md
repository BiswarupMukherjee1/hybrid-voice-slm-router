Device: **GitHub Actions ubuntu-24.04-arm** (Neoverse-N2), 4 CPU threads available. 112 driver turns.

| System | Task success | Resolved | Free phrasing | Multi-intent | Follow-up | Unsafe executed | Model alone refused unsafe | Sent to model | Latency p50 / p95 (s) | First token cold / warm (s) | Decode tok/s | Peak RAM (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Embedded engine only (baseline) | 43% | 28% | 0% | 0% | 0% | 0 | – | 0% | – / – | – / – | – | – |
| Hybrid + gemma3-1b-q4 | 47% | 82% | 43% | 0% | 29% | 4 | 0/12 | 65% | 3.80 / 5.52 | 7.40 / 0.25 | 13.8 | 1579 |
| Hybrid + llama3.2-1b-q4 | 51% | 82% | 22% | 25% | 64% | 4 | 0/12 | 65% | 3.24 / 6.77 | 4.60 / 0.18 | 20.3 | 1872 |
| Hybrid + qwen2.5-0.5b-q4 | 43% | 82% | 17% | 12% | 21% | 4 | 0/12 | 65% | 3.49 / 5.03 | 4.31 / 0.15 | 26.3 | 774 |
| Hybrid + qwen2.5-1.5b-q4 | 71% | 86% | 87% | 62% | 50% | 4 | 0/12 | 65% | 2.95 / 4.60 | 6.15 / 0.24 | 16.2 | 2159 |
| Hybrid + qwen2.5-1.5b-q8 | 75% | 87% | 83% | 75% | 64% | 4 | 0/12 | 65% | 2.74 / 4.47 | 4.74 / 0.21 | 17.4 | 3478 |
| Hybrid + qwen3-1.7b-q4 | 79% | 97% | 83% | 6% | 79% | 0 | 12/12 | 65% | 2.82 / 4.64 | 6.85 / 0.25 | 13.5 | 2762 |

Same model with fewer CPU threads (the language model sharing the CPU with HMI and audio):

| Model | CPU threads | Latency p50 (s) | First token warm p50 (s) | Decode tok/s |
|---|---|---|---|---|
| qwen2.5-1.5b-q4 | 1 | 6.53 | 0.24 | 6.3 |
| qwen2.5-1.5b-q4 | 2 | 4.39 | 0.24 | 10.5 |
| qwen2.5-1.5b-q4 | 4 | 2.95 | 0.24 | 16.2 |

![success vs latency](results/success_vs_latency.png)
