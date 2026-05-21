#!/usr/bin/env python3
"""WCER (Workload-Conditioned Expert Residency) — Stage 1: activation tracer.

Pipeline (see docs/EXPERT_RESIDENCY_AND_PRUNING.md):
  1. expert_trace.py            — trace which experts each token selects (this file)
  2. expert_trace_compare.py    — is routing workload-conditioned? (cross-workload)
  3. expert_resident_manifest.py— derive the per-layer resident set (a "bank")
  4. expert_resident_eval.py    — router-masked quality eval (full weights, no unload)
  5. expert_resident_load.py    — load ONLY the resident set; measure RAM/load/TTFT

Expert activation tracer for MoE models — Track B decision-layer foundation.

The plan's "first useful artifact is an activation-trace format, not a pruning
algorithm" (EXPERT_PARALLELISM_PLAN.md, Track B). This records *which experts
each token selects*, per layer, during normal single-node inference, and emits a
versioned, JSON-shareable summary that resident-set / placement decisions
consume. Requires standard (non-expert-parallel) loading so expert ids are global (the stock
SwitchGLU path passes GLOBAL expert ids; under EP the ids would be rank-local).

Model-agnostic by construction: every MoE model routes through
``block.switch_mlp(x, indices)`` (a fused ``SwitchGLU``), so ``indices`` *is* the
activation trace. We swap each block's ``switch_mlp`` for a transparent wrapper
(an instance swap, NOT an ``__call__`` monkey-patch — Python resolves ``inst(...)``
via ``type(inst).__call__`` and ignores instance-level ``__call__`` attributes, so
only swapping the whole object reliably intercepts the call). Restored on exit.

Lazy-eval note: ``indices`` is a graph node, so the wrapper calls ``mx.eval`` then
copies to numpy before histogramming — correct, at the cost of materializing the
(small) routing subgraph each layer. Profiling runs are not perf benchmarks, so
this is the right tradeoff.

Scope v1: expert IDs only. Router weights are a per-model follow-on (SwitchGLU
doesn't see them — they live in OLMoE's softmax / V4's gate return). IDs are what
hot-expert clustering and resident-set sizing consume.

Run (with $MLX_LM on PYTHONPATH)::

    PYTHONPATH=$MLX_LM python \
        benchmarks/expert_trace.py --self-test          # synthetic, no model
    PYTHONPATH=... python benchmarks/expert_trace.py \
        --model mlx-community/OLMoE-1B-7B-0125-Instruct-4bit \
        --workload general --max-tokens 128 --out /tmp/olmoe-trace.json
"""

from __future__ import annotations

import argparse
import json
import os
from types import SimpleNamespace

import numpy as np
import mlx.core as mx
import mlx.nn as nn

SCHEMA_VERSION = "expert-trace/1"

# Default prompt sets per workload label. Kept tiny + inline so a trace is
# reproducible from the script alone; pass --prompts-file to override.
WORKLOADS = {
    "general": [
        "The capital of France is",
        "Explain why the sky appears blue during the day.",
        "Summarize the plot of a classic novel in two sentences.",
        "Describe the water cycle to a ten-year-old.",
        "What were the main causes of the First World War?",
        "Write a short paragraph about the migration habits of monarch butterflies.",
        "Compare and contrast renewable and non-renewable energy sources.",
        "Give three tips for staying focused while working from home.",
        "Who painted the Mona Lisa and what makes it famous?",
        "Explain the difference between weather and climate.",
    ],
    "code": [
        "Write a Python function that returns the nth Fibonacci number.",
        "def quicksort(arr):",
        "Explain what a hash map is and its average time complexity.",
        "Write a SQL query to find the second-highest salary in an employees table.",
        "Implement a binary search over a sorted list in Python.",
        "What is the difference between a process and a thread?",
        "Refactor this loop into a list comprehension: result = []\nfor x in items:\n    result.append(x*2)",
        "Write a regular expression that matches a valid IPv4 address.",
        "Explain how garbage collection works in a managed runtime.",
        "Write a function to reverse a singly linked list.",
    ],
    "math": [
        "Compute the derivative of x^3 + 2x with respect to x.",
        "If a train travels 60 km in 45 minutes, what is its speed in km/h?",
        "Prove that the square root of 2 is irrational.",
        "Solve the system: 2x + y = 7 and x - y = 2.",
        "What is the probability of rolling two dice and getting a sum of 7?",
        "Find the integral of cos(x) from 0 to pi/2.",
        "Explain the Pythagorean theorem and give an example.",
        "What is the sum of the first 100 positive integers?",
        "Factor the polynomial x^2 - 5x + 6.",
        "Compute the limit of (1 + 1/n)^n as n approaches infinity.",
    ],
}


class _TracingSwitchGLU:
    """Transparent stand-in for a ``SwitchGLU`` that records selected expert ids.

    ``block.switch_mlp`` is replaced by an instance of this; the model's
    ``block.switch_mlp(x, inds)`` then dispatches to this type's ``__call__`` and
    records before delegating to the wrapped module. Inference-only: the wrapped
    weights are reachable via ``.inner`` but not registered as nn parameters
    (tracing is enabled after load, no parameter traversal needed).
    """

    def __init__(self, inner, layer_idx: int, tracer: "ExpertTracer"):
        self.inner = inner
        self._layer_idx = layer_idx
        self._tracer = tracer

    def __call__(self, x, indices, *args, **kwargs):
        self._tracer.record(self._layer_idx, indices)
        return self.inner(x, indices, *args, **kwargs)


class _TracingGate:
    """Transparent stand-in for a logit-returning router gate (nn.Linear). Stashes
    the softmax routing distribution per layer so ``record()`` can accumulate the
    routed (selected) experts' probability mass — the REAP-ish weighted-importance
    signal. Returns the gate output unchanged. Logit gates only (OLMoE/Qwen3); not
    used for V4's MoEGate (returns inds/weights, no logit hook)."""

    def __init__(self, inner, layer_idx: int, tracer: "ExpertTracer"):
        self.inner = inner
        self._layer_idx = layer_idx
        self._tracer = tracer

    def __call__(self, x, *args, **kwargs):
        logits = self.inner(x, *args, **kwargs)
        # Cast to float32 before numpy: bfloat16 (Qwen3 gates) has no numpy dtype
        # and breaks the buffer protocol. Return the ORIGINAL logits unchanged.
        logits32 = logits.astype(mx.float32)
        mx.eval(logits32)
        lg = np.array(logits32).reshape(-1, self._tracer.n_experts).astype(np.float64)
        lg -= lg.max(axis=-1, keepdims=True)
        np.exp(lg, out=lg)
        lg /= lg.sum(axis=-1, keepdims=True)
        self._tracer._gate_probs[self._layer_idx] = lg
        return logits


class ExpertTracer:
    """Accumulates per-layer expert-selection statistics across forward calls."""

    def __init__(self, n_layers: int, n_experts: int, top_k: int):
        self.n_layers = n_layers
        self.n_experts = n_experts
        self.top_k = top_k
        self.hist = np.zeros((n_layers, n_experts), dtype=np.int64)
        self.coact = np.zeros((n_layers, n_experts, n_experts), dtype=np.int64)
        self.prefill_tokens = 0
        self.decode_tokens = 0
        self.calls = 0
        self.hash_layers = []  # layer indices with fixed hash routing (set by enable_tracing)
        # Optional router-weighted importance (REAP-ish): summed softmax routing
        # probability of the SELECTED experts per layer. Populated only when gates
        # are wrapped (logit gates; --capture-weights). _gate_probs is a transient
        # per-layer stash filled by the gate wrapper and consumed in record().
        self.weight_mass = None  # [n_layers, n_experts] or None
        self._gate_probs = {}

    def record(self, layer_idx: int, indices) -> None:
        mx.eval(indices)
        idx = np.array(indices).reshape(-1, self.top_k)  # [n_tokens, top_k]
        n = idx.shape[0]
        # Per-call token count distinguishes prefill (the one multi-token call)
        # from decode (one token per step). Assumes batch size 1.
        if n > 1:
            self.prefill_tokens += n
        else:
            self.decode_tokens += n
        self.hist[layer_idx] += np.bincount(
            idx.reshape(-1), minlength=self.n_experts
        )[: self.n_experts]
        # Unordered co-activation: every pair within a token's top-k set.
        L = self.coact[layer_idx]
        k = idx.shape[1]
        for a in range(k):
            for b in range(a + 1, k):
                np.add.at(L, (idx[:, a], idx[:, b]), 1)
                np.add.at(L, (idx[:, b], idx[:, a]), 1)
        # Router-weighted importance: accumulate the softmax probability of each
        # SELECTED expert (stashed by the gate wrapper for this layer this call).
        probs = self._gate_probs.pop(layer_idx, None)
        if probs is not None:
            if self.weight_mass is None:
                self.weight_mass = np.zeros((self.n_layers, self.n_experts), dtype=np.float64)
            sel_w = probs[np.arange(idx.shape[0])[:, None], idx]  # [n_tokens, top_k]
            np.add.at(self.weight_mass[layer_idx], idx.reshape(-1), sel_w.reshape(-1))
        self.calls += 1

    def summary(self, top_m: int = 16, **meta) -> dict:
        # Coverage / resident-set sizing: how many experts (the hottest ones)
        # cover X% of all token-routes in a layer. This is the number a static
        # resident set is built from.
        cov_pcts = [50, 80, 90, 95, 99]
        cov_points = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        cov_points = [p for p in cov_points if p <= self.n_experts]

        hot, imbalance, coverage_at, experts_for_pct, coact_top = [], [], [], [], []
        for l in range(self.n_layers):
            h = self.hist[l]
            total = int(h.sum())
            order = np.argsort(-h)  # experts hottest-first
            sorted_counts = h[order]
            cum = np.cumsum(sorted_counts) / max(total, 1)

            hot.append([[int(order[i]), int(sorted_counts[i])]
                        for i in range(min(top_m, self.n_experts))])
            mean = total / self.n_experts if self.n_experts else 0
            imbalance.append(float(sorted_counts[0] / mean) if mean else 0.0)
            coverage_at.append({str(p): float(cum[min(p, len(cum)) - 1])
                                for p in cov_points})
            experts_for_pct.append(
                {str(p): int(np.searchsorted(cum, p / 100.0) + 1) for p in cov_pcts}
            )
            # Top co-activated unordered pairs (upper triangle).
            C = np.triu(self.coact[l], k=1)
            if C.max() > 0:
                flat = np.argsort(-C, axis=None)[:top_m]
                pairs = [[int(i), int(j), int(C[i, j])]
                         for i, j in (np.unravel_index(f, C.shape) for f in flat)
                         if C[i, j] > 0]
            else:
                pairs = []
            coact_top.append(pairs)

        weight_mass = (self.weight_mass.tolist() if self.weight_mass is not None else None)
        out = {
            "schema_version": SCHEMA_VERSION,
            "router_weights": ("captured (summed softmax mass of selected experts)"
                               if self.weight_mass is not None
                               else "not_captured (indices-only; pass --capture-weights)"),
            "per_layer_weight_mass": weight_mass,  # router-weighted importance, or null
            "n_layers": self.n_layers,
            "n_experts": self.n_experts,
            "top_k": self.top_k,
            "n_forward_calls": self.calls,
            "hash_layers": self.hash_layers,  # fixed-routing layers (residency = fixed union)
            "prefill_tokens": self.prefill_tokens,
            "decode_tokens": self.decode_tokens,
            "total_routes": int(self.hist.sum()),
            "per_layer_histogram": self.hist.tolist(),
            "hot_experts_per_layer": hot,
            "imbalance_ratio_per_layer": imbalance,
            "coverage_fraction_at_topM_per_layer": coverage_at,
            "experts_needed_for_pct_coverage_per_layer": experts_for_pct,
            "coactivation_top_pairs_per_layer": coact_top,
        }
        out.update(meta)
        return out


# Known attribute names for a layer's MoE block across mlx-lm model families.
# Discovery is the one per-family touch point; the gate/slice/remap are generic.
MOE_BLOCK_ATTRS = ("mlp", "ffn", "block_sparse_moe")  # OLMoE/Qwen3 / V4 / Mixtral


def _find_moe_blocks(model):
    """Return the per-layer MoE blocks (any of MOE_BLOCK_ATTRS w/ switch_mlp)."""
    blocks = []
    for layer in model.model.layers:
        for attr in MOE_BLOCK_ATTRS:
            b = getattr(layer, attr, None)
            if b is not None and hasattr(b, "switch_mlp"):
                blocks.append(b)
                break
    return blocks


def enable_tracing(model, capture_weights: bool = False):
    """Swap each MoE block's ``switch_mlp`` for a recording wrapper.

    With ``capture_weights``, also wrap logit-returning gates (nn.Linear /
    QuantizedLinear — OLMoE/Qwen3) to accumulate router-weighted importance.
    V4's MoEGate is not logit-returning, so weight capture is skipped there.

    Returns ``(tracer, restore)`` — call ``restore()`` to put the originals back.
    """
    blocks = _find_moe_blocks(model)
    if not blocks:
        raise SystemExit("no MoE blocks found (model has no switch_mlp layers)")
    b0 = blocks[0]
    # n_experts from the fused SwitchGLU (ground truth: weight.shape[0]) — robust
    # across families that don't expose block.num_experts (e.g. DeepSeek-V2).
    n_experts = b0.switch_mlp.gate_proj.num_experts
    top_k = getattr(b0, "top_k", None) or b0.num_experts_per_tok
    tracer = ExpertTracer(len(blocks), n_experts, top_k)

    # Record fixed-routing (hash) layers: their residency can only be a fixed
    # union of the hash table's targets, not a dynamic selection (V4 MoEGate).
    tracer.hash_layers = [i for i, b in enumerate(blocks)
                          if getattr(getattr(b, "gate", None), "hash", False)]

    linear_like = (nn.Linear, getattr(nn, "QuantizedLinear", nn.Linear))
    saved, saved_gates = [], []
    for i, b in enumerate(blocks):
        saved.append((b, b.switch_mlp))
        b.switch_mlp = _TracingSwitchGLU(b.switch_mlp, i, tracer)
        if capture_weights and isinstance(getattr(b, "gate", None), linear_like):
            saved_gates.append((b, b.gate))
            b.gate = _TracingGate(b.gate, i, tracer)

    def restore():
        for b, orig in saved:
            b.switch_mlp = orig
        for b, orig in saved_gates:
            b.gate = orig

    return tracer, restore


# --------------------------------------------------------------------------- #
# Self-test: validate the wrapper + accumulation on a synthetic OLMoE block,   #
# no model download. Cross-checks tracer.hist against the block's own routing. #
# --------------------------------------------------------------------------- #

def _self_test() -> None:
    from mlx_lm.models.olmoe import OlmoeSparseMoeBlock

    mx.random.seed(0)
    num_experts, top_k, hidden, inter, tokens = 64, 8, 32, 64, 40
    args = SimpleNamespace(
        num_experts=num_experts, num_experts_per_tok=top_k, norm_topk_prob=True,
        hidden_size=hidden, intermediate_size=inter, mlp_bias=False,
    )
    block = OlmoeSparseMoeBlock(args)
    mx.eval(block.parameters())
    block._ep_enabled = False

    # Independently reproduce the block's routing to get the expected indices.
    x = mx.random.normal((1, tokens, hidden))
    x_flat = x.reshape(-1, hidden)
    rw = mx.softmax(block.gate(x_flat), axis=1, precise=True)
    exp_idx = mx.argpartition(-rw, kth=top_k - 1, axis=-1)[..., :top_k]
    mx.eval(exp_idx)
    expected_hist = np.bincount(np.array(exp_idx).reshape(-1), minlength=num_experts)

    # Wrap a one-layer "model" shim and run the block forward.
    tracer = ExpertTracer(n_layers=1, n_experts=num_experts, top_k=top_k)
    block.switch_mlp = _TracingSwitchGLU(block.switch_mlp, 0, tracer)
    _ = block(x)
    mx.eval(_)

    fired = tracer.calls == 1
    total_ok = tracer.hist.sum() == tokens * top_k
    match = bool(np.array_equal(tracer.hist[0], expected_hist))
    # Co-activation: each token contributes C(top_k,2) unordered pairs, counted
    # symmetrically (x2) in the matrix.
    coact_ok = tracer.coact[0].sum() == tokens * top_k * (top_k - 1)

    print(f"[self-test] wrapper fired:        {fired}")
    print(f"[self-test] total routes == N*k:  {total_ok} ({tracer.hist.sum()} == {tokens*top_k})")
    print(f"[self-test] histogram matches:    {match}")
    print(f"[self-test] coactivation count:   {coact_ok}")
    s = tracer.summary(top_m=4, model_id="synthetic-olmoe", workload_label="self-test")
    print(f"[self-test] summary keys: {sorted(s)[:6]}...")
    print(f"[self-test] layer0 experts_for_90pct = "
          f"{s['experts_needed_for_pct_coverage_per_layer'][0]['90']}/{num_experts}")
    if not (fired and total_ok and match and coact_ok):
        raise SystemExit("SELF-TEST FAILED")
    print("\n[self-test] ALL PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true",
                        help="synthetic validation, no model download")
    parser.add_argument("--model", help="HF repo id of an MoE model")
    parser.add_argument("--workload", default="general", choices=list(WORKLOADS))
    parser.add_argument("--all-workloads", action="store_true",
                        help="load once, trace every workload (separate JSON each)")
    parser.add_argument("--prompts-file", help="newline-delimited prompts (overrides --workload set)")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--prefill-only", action="store_true",
                        help="forward each prompt once (no autoregressive decode) — "
                             "captures prompt-token routing; far faster on big models")
    parser.add_argument("--out", default="/tmp/expert-trace.json",
                        help="output path (single workload); for --all-workloads see --out-dir")
    parser.add_argument("--out-dir", default="/tmp",
                        help="dir for per-workload JSON when --all-workloads")
    parser.add_argument("--top-m", type=int, default=16)
    parser.add_argument("--capture-weights", action="store_true",
                        help="also capture router-weighted importance (logit gates only) "
                             "for a count-vs-weighted selection ablation")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return
    if not args.model:
        raise SystemExit("--model required (or use --self-test)")

    from mlx_lm import load, generate

    model, tokenizer = load(args.model)  # load once
    slug = args.model.split("/")[-1]

    def trace_workload(label, prompts, out_path):
        prompts = [p for p in prompts if p.strip()]
        tracer, restore = enable_tracing(model, capture_weights=args.capture_weights)
        try:
            for p in prompts:
                if args.prefill_only:
                    mx.eval(model(mx.array([tokenizer.encode(p)])))
                else:
                    generate(model, tokenizer, prompt=p, max_tokens=args.max_tokens)
        finally:
            restore()
        summary = tracer.summary(
            top_m=args.top_m, model_id=args.model, workload_label=label,
            n_prompts=len(prompts), max_tokens_per_prompt=args.max_tokens,
            prefill_only=args.prefill_only,
        )
        json.dump(summary, open(out_path, "w"), indent=2)
        imb = summary["imbalance_ratio_per_layer"]
        e90 = [d["90"] for d in summary["experts_needed_for_pct_coverage_per_layer"]]
        hl = summary["hash_layers"]
        sl = [l for l in range(summary["n_layers"]) if l not in hl]  # score-routed
        def mean_e90(idxs):
            return sum(e90[i] for i in idxs) / len(idxs) if idxs else 0.0
        print(f"[expert-trace] {slug} workload={label} layers={summary['n_layers']} "
              f"experts={summary['n_experts']} top_k={summary['top_k']} "
              f"hash_layers={hl} prefill_only={args.prefill_only}")
        print(f"[expert-trace]  routes total={summary['total_routes']} "
              f"(prefill={summary['prefill_tokens']} decode={summary['decode_tokens']})")
        print(f"[expert-trace]  imbalance(max/mean) min={min(imb):.2f} max={max(imb):.2f} "
              f"mean={sum(imb)/len(imb):.2f}")
        print(f"[expert-trace]  experts for 90%% cov: score-layers mean={mean_e90(sl):.1f}  "
              f"hash-layers mean={mean_e90(hl):.1f}  / {summary['n_experts']}")
        print(f"[expert-trace]  wrote {out_path}")

    if args.all_workloads:
        for label, prompts in WORKLOADS.items():
            trace_workload(label, prompts, os.path.join(args.out_dir, f"{slug}-trace-{label}.json"))
    else:
        prompts = (open(args.prompts_file).read().splitlines()
                   if args.prompts_file else WORKLOADS[args.workload])
        trace_workload(args.workload, prompts, args.out)


if __name__ == "__main__":
    main()
