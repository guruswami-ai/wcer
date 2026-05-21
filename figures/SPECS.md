# WCER figures — generation specs

Four infographics explain the paper visually. Generated with a text-to-image model
(Google Gemini / Imagen, or any capable backend). Suggested style across all four:
**clean McKinsey/BCG-style technical infographic, dark navy gradient background,
indigo + amber accents, green to mark savings, no neon/cyberpunk, legible labels.**
Drop the rendered PNG/JPG next to this file and reference it from `WCER_DRAFT.md`.

> Status: pending an available image backend. The two data-driven figures (2 and the
> Pareto) are ideally rendered as real charts (matplotlib) from the numbers below;
> figures 1, 3, 4 are illustrative.

---

## Figure 1 — The WCER concept (comparison)

**Purpose:** the one-glance idea — load only the experts a workload uses.

**Prompt:**
> Two side-by-side panels comparing memory use in a Mixture-of-Experts model.
> LEFT, "Standard loading": a grid of 16 expert blocks, ALL lit/loaded, caption
> "all experts resident — most unused". RIGHT, "WCER": the same grid but only ~5
> blocks highlighted (the workload's experts), the rest greyed and labeled "cold
> (on disk)", with a router arrow routing tokens only to the highlighted experts,
> caption "only the workload's experts resident → less memory, identical answers".
> Bottom strip: "Keep only the experts your workload actually uses."

## Figure 2 — Concentration predicts savings (data chart)

**Purpose:** the headline scientific result — read the payoff off the trace.

**Render as a chart** (x = "experts needed for 90% of routing", y = "memory cut at full quality"), 5 labeled points, a trend line, an annotation "measurable from one trace, before deploying":

| model | experts for 90% of routing | memory cut |
|---|---|---|
| Mixtral-8x7B | 88% | 14% |
| OLMoE-1B-7B | 72% | 23% |
| DeepSeek-V2-Lite | 66% | 23% |
| Qwen3-30B-A3B | 38% | 47% |
| DeepSeek-V4-Flash | 25% | 68% |

Caption: "The more concentrated a model's routing (fewer experts cover its work), the more WCER saves — and you can measure this before you deploy."

## Figure 3 — Use cases (ecosystem / mindmap)

**Purpose:** where it's useful, simply.

**Prompt:**
> A central node "WCER: load only the experts you need" with four branches:
> (1) "Per-workload servers" — three small server icons labeled code / math / chat,
> each holding a different small subset of experts; (2) "Fit a bigger MoE on a
> smaller machine" — a large model icon shrinking to fit a small box; (3) "Keep
> precision, don't over-quantize" — a dial showing '8-bit, fewer experts' vs
> '4-bit, all experts'; (4) "Predict before you spend" — a magnifying glass over a
> trace with a green check / red cross. Clean iconographic infographic.

## Figure 4 — FORGE: the research direction (flowchart)

**Purpose:** the forward vision WCER opens.

**Prompt:**
> A left-to-right pipeline flowchart titled "FORGE (research direction)".
> Stage 1 "Real workload traffic" (not synthetic) → Stage 2 "Trace which experts
> matter" (the WCER trace, highlighted as the validated foundation) → Stage 3
> "Joint search: which experts resident × at what precision" (a grid of
> residency-vs-quantization options with a Pareto frontier) → Stage 4, three
> outputs branching: "Resident model (fits this machine)", "Trace-driven pruning",
> "Trace-driven fine-tuning". A side note: "conditioned on a specific machine and a
> specific workload." Mark Stage 2 (WCER) as done/validated and Stages 3–4 as
> research direction.
