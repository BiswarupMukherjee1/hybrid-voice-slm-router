Device: **GitHub Actions ubuntu-24.04-arm** (Neoverse-N2), 4 CPU threads available. 10 driver turns.

| System | Task success | Resolved | Free phrasing | Multi-intent | Follow-up | Unsafe executed | Model alone refused unsafe | Sent to model | Latency p50 / p95 (s) | First token cold / warm (s) | Decode tok/s | Peak RAM (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Embedded engine only (baseline) | 100% | 100% | – | – | – | 0 | – | 0% | – / – | – / – | – | – |
| Hybrid + gemma3-1b-q4 | 100% | 100% | – | – | – | 0 | 0/0 | 0% | – / – | – / – | – | 1424 |
| Hybrid + llama3.2-1b-q4 | 100% | 100% | – | – | – | 0 | 0/0 | 0% | – / – | – / – | – | 1756 |
| Hybrid + qwen2.5-0.5b-q4 | 100% | 100% | – | – | – | 0 | 0/0 | 0% | – / – | – / – | – | 1409 |
| Hybrid + qwen2.5-1.5b-q4 | 100% | 100% | – | – | – | 0 | 0/0 | 0% | – / – | – / – | – | 2107 |
| Hybrid + qwen2.5-1.5b-q8 | 100% | 100% | – | – | – | 0 | 0/0 | 0% | – / – | – / – | – | 3476 |
| Hybrid + qwen3-1.7b-q4 | 100% | 100% | – | – | – | 0 | 0/0 | 0% | – / – | – / – | – | 2648 |

Same model with fewer CPU threads (the language model sharing the CPU with HMI and audio):

| Model | CPU threads | Latency p50 (s) | First token warm p50 (s) | Decode tok/s |
|---|---|---|---|---|
| qwen2.5-1.5b-q4 | 1 | – | – | – |
| qwen2.5-1.5b-q4 | 2 | – | – | – |
| qwen2.5-1.5b-q4 | 4 | – | – | – |
