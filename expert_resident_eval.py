#!/usr/bin/env python3
"""WCER Stage 4 — router-masked quality eval (full weights, no unload).

Measures the QUALITY cost of a resident set by router-masking the full model
to it (so non-resident experts are never selected) and comparing perplexity in-
and out-of-domain. Memory is reported analytically here; Stage 5
(expert_resident_load.py) does the real unload + RAM/load/TTFT measurement.
(See README.md.)

Resident-set quality/memory evaluation — WCER stage (close the loop).

Builds a static resident set from an expert-trace (the hottest experts per
layer), restricts the model's router to it (router-masking: non-resident expert
logits -> -inf, so the top-k can only pick resident experts), and measures the
QUALITY cost against the full model — the plan's hard rule: never prune without
a full-model reference.

Headline: perplexity full vs. router-masked, evaluated both IN-distribution and
OUT-of-distribution relative to the workload the resident set was tuned on. That
ties the coverage gap from expert_trace_compare.py to an actual quality number:
a general-tuned set should hold perplexity on general text but degrade on code.

Router-masking measures the *quality* cost of a resident set. The *memory* saving
is reported analytically (resident fraction x measured expert-weight share of the
model); actually unloading non-resident experts (real RAM/load-time win) is the
follow-on (plan step 2 "loaded resident set"). OLMoE-style gates only for now
(gate is an nn.Linear returning expert logits); V4's MoEGate is a per-model
follow-on (hash layers aren't logit-maskable the same way).

Run (with $MLX_LM on PYTHONPATH)::

    PYTHONPATH=$MLX_LM python \
        benchmarks/expert_resident_eval.py \
        --model mlx-community/OLMoE-1B-7B-0125-Instruct-4bit \
        --trace /tmp/olmoe-trace-general.json --budgets 0.25 0.5 0.75
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
import mlx.core as mx
import mlx.nn as nn

NEG_INF = -1e9

# Held-out eval passages — deliberately DISTINCT from the trace prompts
# (WORKLOADS in expert_trace.py), so the resident set is never evaluated on the
# text it was built from. Several passages per domain (token-weighted aggregate
# perplexity) to cut single-passage variance.
EVAL_TEXT = {
    "general": [
        "The history of written language spans thousands of years, beginning with "
        "pictographic systems and evolving into the alphabets used today. Scholars "
        "study how meaning, grammar, and script changed across cultures and eras.",
        "Ocean currents redistribute heat around the planet, moderating coastal "
        "climates and shaping weather patterns far inland. The Gulf Stream, for "
        "instance, carries warm water northward across the Atlantic.",
        "A balanced diet provides the body with carbohydrates, proteins, fats, "
        "vitamins, and minerals in proportions that support growth, repair, and "
        "the steady supply of energy needed throughout an ordinary day.",
        "The printing press transformed European society by making books cheaper "
        "and more plentiful, accelerating the spread of literacy, scientific ideas, "
        "and political pamphlets across borders within a few generations.",
        "Photosynthesis converts sunlight, water, and carbon dioxide into glucose "
        "and oxygen. It underpins almost every food chain and is responsible for "
        "the oxygen that most living organisms depend on to breathe.",
    ],
    "code": [
        "import collections\n\n"
        "def group_anagrams(words):\n"
        "    buckets = collections.defaultdict(list)\n"
        "    for w in words:\n"
        "        key = ''.join(sorted(w))\n"
        "        buckets[key].append(w)\n"
        "    return list(buckets.values())\n",
        "def binary_search(arr, target):\n"
        "    lo, hi = 0, len(arr) - 1\n"
        "    while lo <= hi:\n"
        "        mid = (lo + hi) // 2\n"
        "        if arr[mid] == target:\n"
        "            return mid\n"
        "        elif arr[mid] < target:\n"
        "            lo = mid + 1\n"
        "        else:\n"
        "            hi = mid - 1\n"
        "    return -1\n",
        "class Stack:\n"
        "    def __init__(self):\n"
        "        self._items = []\n"
        "    def push(self, x):\n"
        "        self._items.append(x)\n"
        "    def pop(self):\n"
        "        return self._items.pop()\n"
        "    def is_empty(self):\n"
        "        return not self._items\n",
        "async def fetch_all(urls, session):\n"
        "    tasks = [session.get(u) for u in urls]\n"
        "    responses = await asyncio.gather(*tasks)\n"
        "    return [await r.text() for r in responses]\n",
        "SELECT department, COUNT(*) AS headcount, AVG(salary) AS avg_salary\n"
        "FROM employees\n"
        "WHERE active = 1\n"
        "GROUP BY department\n"
        "HAVING COUNT(*) > 5\n"
        "ORDER BY avg_salary DESC;\n",
    ],
}


class _MaskedGate:
    """OLMoE gate stand-in that forces routing onto a resident expert set."""

    def __init__(self, inner: nn.Linear, resident_mask: mx.array):
        self.inner = inner
        self.mask = resident_mask.reshape(1, -1)  # [1, n_experts] bool

    def __call__(self, x):
        logits = self.inner(x)
        return mx.where(self.mask, logits, mx.array(NEG_INF, dtype=logits.dtype))


def _moe_blocks(model):
    blocks = []
    for layer in model.model.layers:
        for attr in ("mlp", "ffn", "block_sparse_moe"):  # OLMoE/Qwen3 / V4 / Mixtral
            b = getattr(layer, attr, None)
            if b is not None and hasattr(b, "switch_mlp"):
                blocks.append(b)
                break
    return blocks


def build_resident_masks(trace_path, budget_frac):
    d = json.load(open(trace_path))
    hist = np.array(d["per_layer_histogram"], dtype=np.float64)  # [L, E]
    n_layers, n_experts = hist.shape
    budget = max(1, int(round(budget_frac * n_experts)))
    masks = np.zeros((n_layers, n_experts), dtype=bool)
    for l in range(n_layers):
        masks[l, np.argsort(-hist[l])[:budget]] = True
    return masks, budget, n_experts, d.get("top_k")


def enable_masking(model, masks):
    blocks = _moe_blocks(model)
    if len(blocks) != masks.shape[0]:
        raise SystemExit(f"layer mismatch: {len(blocks)} blocks vs {masks.shape[0]} mask rows")
    # OLMoE's router is a (possibly quantized) Linear returning expert logits;
    # V4's MoEGate returns (inds, weights) and includes hash layers — masking
    # that is a per-model follow-on, so accept only logit-returning gates.
    linear_like = (nn.Linear, getattr(nn, "QuantizedLinear", nn.Linear))
    saved = []
    for i, b in enumerate(blocks):
        if not isinstance(b.gate, linear_like):
            raise SystemExit(f"router-masking supports logit-returning gates only; "
                             f"got {type(b.gate).__name__} (V4 MoEGate is a follow-on)")
        saved.append((b, b.gate))
        b.gate = _MaskedGate(b.gate, mx.array(masks[i]))

    def restore():
        for b, orig in saved:
            b.gate = orig

    return restore


def perplexity(model, tokenizer, text) -> float:
    """Token-weighted aggregate perplexity over one or several held-out passages.

    A list reduces the single-passage variance that jags the budget→quality
    curve; aggregating NLL over all tokens (not averaging per-passage ppl) is the
    standard corpus perplexity.
    """
    passages = [text] if isinstance(text, str) else list(text)
    total_nll, total_tok = 0.0, 0
    for p in passages:
        ids = mx.array([tokenizer.encode(p)])
        logits = model(ids).astype(mx.float32)[0]          # [T, V]
        logp = logits[:-1] - mx.logsumexp(logits[:-1], axis=-1, keepdims=True)
        nll_sum = -mx.take_along_axis(logp, ids[0, 1:, None], axis=-1).sum()
        mx.eval(nll_sum)
        total_nll += float(nll_sum.item())
        total_tok += ids.shape[1] - 1
    return math.exp(total_nll / max(total_tok, 1))


def expert_weight_fraction(model) -> float:
    """Share of model parameter elements that live in the routed experts."""
    def count(tree):
        if isinstance(tree, mx.array):
            return tree.size
        if isinstance(tree, dict):
            return sum(count(v) for v in tree.values())
        if isinstance(tree, list):
            return sum(count(v) for v in tree)
        return 0
    total = count(model.parameters())
    experts = sum(count(b.switch_mlp.parameters()) for b in _moe_blocks(model))
    return experts / total if total else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/OLMoE-1B-7B-0125-Instruct-4bit")
    ap.add_argument("--trace", required=True, help="expert-trace JSON for the resident set")
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    ap.add_argument("--out", default="/tmp/resident-eval.json")
    args = ap.parse_args()

    from mlx_lm import load
    model, tokenizer = load(args.model)
    exp_frac = expert_weight_fraction(model)

    # Full-model reference perplexity (the baseline every resident set is judged against).
    full = {dom: perplexity(model, tokenizer, txt) for dom, txt in EVAL_TEXT.items()}
    src = json.load(open(args.trace)).get("workload_label", "?")

    rows = []
    for bf in sorted(args.budgets):
        masks, budget, n_experts, _ = build_resident_masks(args.trace, bf)
        restore = enable_masking(model, masks)
        try:
            masked = {dom: perplexity(model, tokenizer, txt) for dom, txt in EVAL_TEXT.items()}
        finally:
            restore()
        mem_saving = (1 - budget / n_experts) * exp_frac
        rows.append({"budget_frac": budget / n_experts, "budget": budget, "n_experts": n_experts,
                     "masked_ppl": masked, "expert_mem_saving_frac": mem_saving})

    result = {
        "model": args.model, "resident_set_from": src, "trace": args.trace,
        "expert_weight_fraction": exp_frac, "full_ppl": full, "budgets": rows,
        "eval_domains": list(EVAL_TEXT),
    }
    json.dump(result, open(args.out, "w"), indent=2)

    domains = list(EVAL_TEXT)
    print(f"Model: {args.model}")
    print(f"Resident set tuned on: '{src}'.  Experts = {exp_frac*100:.0f}% of model weights.\n")
    print(f"FULL-MODEL perplexity:  " + "  ".join(f"{d}={full[d]:.2f}" for d in domains))
    print(f"\nROUTER-MASKED to resident set (ppl, and ratio vs full):")
    hdr = "  budget  mem_save" + "".join(f"{d:>16}" for d in domains)
    print(hdr)
    for r in rows:
        cells = ""
        for d in domains:
            ratio = r["masked_ppl"][d] / full[d]
            cells += f"  {r['masked_ppl'][d]:7.2f} (x{ratio:.2f})"
        print(f"  {r['budget']:>2}/{r['n_experts']:<3} {r['expert_mem_saving_frac']*100:5.0f}%" + cells)
    print(f"\n(in-domain = '{src}'; degradation should be smaller in-domain than out — "
          f"the quality cost of a workload-mismatched resident set.)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
