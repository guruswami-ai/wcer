# WCER results — every headline number + how it was produced

All runs: 4-bit MLX checkpoints, Apple Silicon (DeepSeek-V4-Flash on Mac Studio M3 Ultra; the rest on Mac Studio / Mac Mini M4 Pro), single-node, mlx-lm on `PYTHONPATH` (DeepSeek models patched with `patches/`). Quality = token-weighted aggregate perplexity over held-out passages (`expert_resident_eval.EVAL_TEXT`), deliberately distinct from the trace prompts. "Behavior-preserving" = `expert_resident_load.py --mode check`: `top1_match=1.000` (in practice `rel=0.0`).

## 1. Concentration → savings (5 models, 4 families)

Trace each model (`expert_trace.py --all-workloads --prefill-only`); read `experts_needed_for_pct_coverage_per_layer["90"]`; run the budget Pareto (`wcer_search.py`).

| model | family | imbalance (max/mean) | experts for 90% cov | usable budget | RAM cut |
|---|---|---|---|---|---|
| Mixtral-8x7B-Instruct-v0.1-4bit | Mistral | ~1.5 | 88% (7/8) | ~88% | ~14% |
| OLMoE-1B-7B-0125-Instruct-4bit | OLMo | ~3.5 | 72% | ~75% | ~23% |
| DeepSeek-V2-Lite-Chat-4bit-mlx | DeepSeek | ~5 | 66% | ~75% | ~23% |
| Qwen3-30B-A3B-mixed-3-4bit | Qwen | ~10 | 38% | ~50% | ~47% |
| DeepSeek-V4-Flash-4bit | DeepSeek | ~35 | 25% | ~25% | **~68%** |

DeepSeek-V4-Flash headline (full-model run, M3 Ultra): full **160.4 GB** peak / TTFT 1080 ms / general ppl 30.29; resident @25% **51.7 GB** (−68%) / TTFT **348 ms** / ppl ×1.21; `check` PASS.

## 2. Qwen3 Pareto (multi-passage held-out perplexity, weighted selection)

`wcer_search.py --model …Qwen3… --trace …general… --budgets 0.25 0.4 0.5 0.6 0.75 --selection weighted`

| budget | peak RAM | TTFT | in-domain ppl ×full |
|---|---|---|---|
| full | 14.3 GB | 83 ms | ×1.00 |
| 25% | 4.4 GB | 59 ms | ×4.29 (over tol) |
| 50% | **7.7 GB** | 52 ms | **×0.99** (Pareto) |
| 75% | 11.0 GB | 59 ms | ×0.91 (Pareto) |

## 3. Selection: router-weighted beats count (`--selection weighted` vs `count`, 50%)

| model | count | weighted | full |
|---|---|---|---|
| Qwen3-30B-A3B | 18.94 | **15.20** | 15.19 |
| OLMoE-1B-7B | 51.36 | **42.41** | 15.87 |

Trace-driven (any) selection beats **random** (`manifest --random`) only where routing is concentrated — Qwen3 50% weighted 15.2 vs random ~2948; OLMoE no clear win.

## 4. Shared-expert / random-safety axis (`manifest --random`, 50%, in-domain ppl)

| model | shared expert | random degradation |
|---|---|---|
| OLMoE-1B-7B | no | catastrophic |
| Qwen3-30B-A3B | no | ×194 |
| DeepSeek-V2-Lite | yes | **×2.4 (graceful)** |
| DeepSeek-V4-Flash | yes | ×2.5 (graceful) |

## 5. Operational costs (Qwen3 unless noted)

- **Bank-switch** (`--mode switch`, general→code @50%): cold re-materialize 3.0 s; Jaccard(A,B)=0.53 → warm incremental ~1.4 s (≈47% of a bank).
- **Decode TPS** (`--mode tps`, batch-1): full 84.0 vs resident@50% 83.2 tok/s — neutral, at half the RAM. Memory/TTFT lever, not throughput.
- **Cold-disk load** (`--mode full/resident` between `sudo purge`): cold full 2.87 s vs cold resident-50% 3.38 s — **not improved** (lazy gather reads full rows). Warm wins are materialize/dequant time.

## 6. Dense baseline (~8 GB matched RAM)

`expert_resident_load.py --mode full/tps` on a dense model vs WCER'd MoE.

| | RAM | general ppl | decode tok/s |
|---|---|---|---|
| dense Qwen3-14B-4bit | 8.4 GB | **10.96** | 26.5 |
| WCER'd Qwen3-30B-A3B @50% | 7.7 GB | 16.41 | **83.2 (3.1×)** |

Quality↔throughput tradeoff (dense = more active params = higher quality; WCER'd-MoE = fewer active params = faster). WCER makes a *chosen* MoE fit; it is not "best quality per GB."

## Notes / caveats (full list in `WCER_DRAFT.md` §4)

- Load/TTFT times are **warm-cache** materialize, not cold-disk reads.
- Perplexity is single-corpus held-out; task-accuracy + more workloads remain.
- V4-Flash autoregressive *generation* tracing crashes on our build (~12 s/tok) → traces are prefill-only (sufficient for a routing map).
- Apple Silicon / MLX only; non-Apple is a hypothesis (see §6 of the draft).
