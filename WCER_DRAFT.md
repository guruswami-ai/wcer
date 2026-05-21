# Workload-Conditioned Expert Residency (WCER)

### Run only the experts your workload actually uses

**Status:** Draft for discussion / paper. Tested on 5 Mixture-of-Experts models from 4 families on Apple Silicon (MLX). Numbers and exact commands: `RESULTS.md`.

---

## In short

**We found a way to run large MoE models with less memory by keeping only the experts a workload actually uses.**

**What we did**
- Built a trace system that watches which experts get used for real prompts.
- Used the trace to load *only* the useful experts into memory.
- Proved the slimmed model behaves identically to the full model under the same routing mask (a hard correctness check, not an approximation).
- Measured *when* this helps and when it doesn't — across 5 models / 4 families.
- Found a clear pattern: on **balanced** MoEs the gain is small (~14%); on **concentrated** MoEs it's large (up to ~68% memory cut, no quality loss).
- Confirmed the tradeoff is real: it saves **memory** and improves **warm load time + time-to-first-word**, but does **not** improve cold-from-disk load (current loader) and is usually a **memory/throughput tradeoff**, not a free win.

## Visual summary

![Concentration predicts savings](figures/concentration-vs-savings.png)
*One trace predicts the payoff before you deploy.*

![Where WCER is useful](figures/wcer-use-cases.jpg)
*Per-workload servers, bigger MoEs on smaller machines, and less over-quantization.*

![FORGE](figures/forge-flowchart.jpg)
*Trace-driven residency is the foundation for a broader residency × quantization search.*

**Why it matters**
- A model can sometimes fit on a machine that would otherwise be too small.
- You can run a larger or higher-precision MoE without forcing aggressive quantization.
- You can tailor a model to a specific workload — code, chat, or math.
- It turns expert pruning from a blunt offline idea into a *runtime residency* strategy.
- You can **predict beforehand** whether it's worth trying, from how concentrated the routing is.

**The short version:** we didn't just find a new way to prune models — we found a way to make MoE models *fit and serve better for specific workloads*, and we learned *when that's worth doing*. That makes it a real deployment tool, not just a benchmark result.

---

## The idea in one minute

A **Mixture-of-Experts (MoE)** model is built from many small sub-networks called **experts**. For each word it processes, a *router* picks just a few experts to do the work and ignores the rest. A model might hold 256 experts but use only 6 per word.

The catch: even though it only *uses* a few at a time, it normally has to *load all of them* into memory, because in principle any word could need any expert.

**WCER asks a practical question:** if a machine only ever does one kind of job — say, answering coding questions — does it really need all 256 experts loaded? Or do coding questions reliably use the same ~60, so the other ~200 can stay on disk?

If so, you can **load only the experts your workload uses, keep the rest cold, and save a lot of memory — without changing the model's answers.**

### An analogy

Think of the experts as **specialist departments in a large company**. Any given project only needs a few departments. If your office *only* handles, say, legal contracts, you don't need the engineering, marketing, and sales teams physically in the building — you can save office space by keeping only the departments your work actually uses.

WCER does this for an MoE model:
- **Experts = departments.** **Resident set = who's in the office today.**
- **Workload = the kind of work** (coding, math, chat). Different work uses different departments.
- It works **only if your work reliably uses a small set of departments** (some models spread work evenly across all of them — for those, this saves nothing).
- A **"shared expert"** (some models have one) is like a **generalist who helps on everything** — so even if you guess the specialists slightly wrong, the generalist keeps quality from collapsing.

The key safety property: **the slimmed-down model gives the same answers as the full model would if you'd simply told it to only use those experts.** It's a memory optimization, not a behavior change — we verify this every time (identical next-word choices).

---

## When it helps — and when it doesn't

WCER's benefit depends on **how concentrated the routing is** — and you can measure this from a quick trace *before* committing to anything.

- ✅ **Helps a lot** when a workload reliably uses a *small fraction* of the experts (concentrated routing). Example: one model we tested needs only 25% of its experts to handle 90% of its work → we cut its memory **68%** with no quality loss.
- ⚠️ **Helps modestly** for moderately-concentrated models (~20–25% memory savings).
- ❌ **Doesn't help** for "load-balanced" models that spread work evenly across all experts (e.g. an 8-expert model that needs 7 of 8 — only ~14% savings, not worth it).

**You can predict which case you're in by tracing the model once.** That's the most useful practical result: WCER tells you in advance whether it's worth deploying.

---

## Use cases (plain)

**1. A dedicated server for one kind of work.**
A coding-assistant endpoint keeps only the experts coding uses; a math endpoint keeps the math experts; a chat endpoint keeps the chat experts. Each fits in **less memory** and starts answering **faster** (lower time-to-first-word) — because it loaded fewer experts. When the workload changes, you swap the "bank" of experts (takes ~1–3 seconds, so it's a per-shift change, not per-request).

**2. Fit a specific large MoE on a smaller machine.**
If you *want* a particular big MoE — for its speed or capabilities — but it doesn't fit your RAM/VRAM, WCER can make it fit by keeping only the experts your workload needs. (Important caveat below: this is about fitting *the model you want*, not getting the best possible quality for your memory.)

**3. Keep more precision instead of crushing the whole model (hypothesis).**
Normally, to squeeze a model into a fixed memory budget you quantize it heavily (e.g. down to 4-bit), which costs quality. WCER offers another knob: keep *fewer experts* but at *higher precision*. On a fixed-VRAM GPU this could let you run a model at, say, 8-bit that wouldn't otherwise fit — trading "fewer experts" for "more precision." (We've shown the memory mechanism works on Apple Silicon; the GPU version isn't built yet.)

**4. Decide what to permanently prune later.**
The same usage traces tell you which experts a workload *never* touches — useful input if you later want to permanently slim the model (pruning). WCER is the reversible, runtime version; permanent pruning (e.g. REAP-style — see the published `guruswami1/Viveka-GLM-4.7-23B-REAP-Smarty-MLX`) is the irreversible version. WCER and REAP consume the same kind of usage signal but answer different questions: REAP asks *which experts to remove from the checkpoint*; WCER asks *which experts to keep resident right now for this workload* (and its traces can feed a later REAP-style decision).

**5. Know before you spend.**
Trace a model, read its concentration, and predict the savings — including predicting that it *won't* help — before downloading 100s of GB or standing up a node.

**When NOT to use it:** load-balanced models (no savings); when a smaller *dense* (non-MoE) model that fits the same memory would give better quality (see the tradeoff below); or when the workload is unpredictable and constantly mixed (a wrong expert set hurts more than no optimization).

---

## What we tested and found

We ran WCER on **5 MoE models from 4 different families** (Mixtral, OLMoE, Qwen3, two DeepSeek models), at 4-bit, on Apple Silicon.

### Finding 1 — the savings are predictable from a trace

"90%-coverage" = how many experts you need to cover 90% of the work. The lower it is, the more concentrated the model, the more you save. This held cleanly across all five:

| model | family | experts for 90% of work | memory cut (no quality loss) |
|---|---|---|---|
| Mixtral-8x7B | Mistral | 88% (7 of 8) | ~14% (not worth it) |
| OLMoE-1B-7B | OLMo | 72% | ~23% |
| DeepSeek-V2-Lite | DeepSeek | 66% | ~23% |
| Qwen3-30B-A3B | Qwen | 38% | ~47% |
| DeepSeek-V4-Flash | DeepSeek | 25% | **~68%** (160→52 GB) |

**Takeaway:** one cheap trace tells you the payoff. Note concentration isn't a "brand" thing — the two DeepSeek models are far apart, so you can't assume; you measure.

### Finding 2 — a "generalist" expert is a safety net

Some models have a **shared expert** that runs for every word. We found this changes what happens if you pick the *wrong* experts to keep: models *with* a shared expert degrade **gracefully** (quality dips a little), while models *without* one degrade **catastrophically** (quality falls off a cliff). So the shared expert is a cushion against a bad guess.

### Finding 3 — pick experts by importance, not just frequency

Choosing which experts to keep by *how strongly* the router prefers them (not just how *often* it picks them) gives better quality — on one model it recovered essentially full quality at half the experts.

### Finding 4 — it saves memory and start-up latency, not raw speed

Keeping fewer experts **halves the memory** and **cuts time-to-first-word ~3×**, but it does **not** change the model's words-per-second once it's running (the model still does the same per-word work). So WCER is a *memory and responsiveness* tool, not a throughput tool.

### Finding 5 — the honest comparison: it's a tradeoff, not a free win

The toughest question: is a slimmed-down big MoE better than just running a smaller *dense* model that fits the same memory? At ~8 GB:

| | quality (lower = better) | speed |
|---|---|---|
| a dense 14B model | **10.96** (better) | 26.5 words/sec |
| WCER'd 30B MoE @ half its experts | 16.41 | **83.2 words/sec (3.1× faster)** |

**The dense model gives better quality; the slimmed MoE is 3× faster.** So WCER is **not** "the best quality for your memory." It's a way to run *a specific MoE you want* (for its speed or abilities) in less memory, while keeping its behavior intact. Choose it when you want that model — not as a blanket replacement for a dense model.

---

## Honest limits

- **Tested on Apple Silicon / MLX only.** A version for NVIDIA GPUs is a plausible next step, not a proven result.
- **Doesn't speed up cold start-from-disk** (it still reads all expert data from disk; it saves *memory* and *warm* start time, not disk reads).
- **Quality measured by perplexity** on held-out text; broader task testing remains.
- **Best for stable, classifiable workloads.** A wrong expert set is worse than none.
- **Not for non-MoE (dense) models** — there's nothing to leave out.

---

## How it works (for the technically curious)

Five steps, mostly model-agnostic: **(1)** trace which experts each word uses; **(2)** check whether different workloads use different experts; **(3)** build a per-layer "resident set" (the experts to keep); **(4)** measure quality of the full model restricted to that set; **(5)** load *only* those experts, restrict the router to them, and serve. Step 5's output is verified identical to step 4's full model — that's the safety guarantee. Adapting a new model family is usually zero or ~6 lines of code (only the router-restriction differs).

---

## Bottom line

> **WCER lets you run a Mixture-of-Experts model using only the experts your workload actually needs — saving memory and start-up time without changing the model's answers. How much it helps is set by how concentrated the model's routing is, which you can measure from a quick trace before deploying. It is a way to make *a chosen model fit*, not a way to get the best possible quality for a given memory budget — for that, a dense model that fits may win. Demonstrated across four model families as a measurable, bounded, honest mechanism.**

*Optional headline follow-up: the same approach at the largest scale (DeepSeek-V4-Pro) and on fixed-VRAM GPUs — both deferred until the core result is written up.*
