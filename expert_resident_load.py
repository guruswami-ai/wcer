#!/usr/bin/env python3
"""WCER Stage 5 — resident-set loader/unloader + measurement (the real feature).

The core of Workload-Conditioned Expert Residency: load ONLY the experts a
workload needs (per the Stage-3 manifest), keeping cold experts off-device, and
measure the real RAM / load / TTFT win — validated behavior-preserving against
the full model under the same router mask. (See docs/EXPERT_RESIDENCY_AND_PRUNING.md.)

Resident-set loader/unloader + measurement — Track B step 2 (the real feature).

Manifest-driven: loads a model with ONLY the resident experts materialized
(lazy load -> fancy-index each switch_mlp's expert arrays to the manifest's
global ids -> eval pages in just those rows from the mmap'd safetensors), and
installs router-masking (non-resident logits -> -inf) + a global->local index
remap so the shortened expert bank is addressed correctly. Non-expert params
(embed/attn/norm/lm_head) load normally.

This is behavior-preserving by construction: the resident-loaded model and the
full model router-masked to the SAME set make identical routing decisions and
run the identical resident experts, so their logits agree to fp-reordering
(top1 exact, rel < 5e-3 — NOT bit-identical: changing the expert-array length
reorders quantized matmul ops). `--mode check` proves it. The win is real RAM +
cold-start materialization, not a behavior change.

Modes (run each in its own process for clean process-isolated peak memory):
  full                    reference: full model, no mask. Peak mem + TTFT + ppl baseline.
  resident --manifest M   unloaded model. Peak mem + load time + TTFT + resident ppl.
  check    --manifest M   masked-full vs resident logits diff (the invariant) + both ppls.

Cold-start load time is reported but is WARM-CACHE here (the safetensors are in
the OS page cache from prior runs); `sudo purge` between runs gives true cold.
Peak memory is process-isolated and clean. The materialization compute (dequant
of fewer experts) saving shows up in eval/TTFT time regardless of cache.

Run::
    PYTHONPATH=$MLX_LM python benchmarks/expert_resident_load.py \
        --model mlx-community/OLMoE-1B-7B-0125-Instruct-4bit --mode full
    ... --mode resident --manifest /tmp/olmoe-resident-general-50.json
    ... --mode check    --manifest /tmp/olmoe-resident-general-50.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expert_resident_eval import EVAL_TEXT, perplexity  # noqa: E402

NEG_INF = -1e9


class _MaskedGate:
    """Force routing onto the resident set: non-resident expert logits -> -inf."""

    def __init__(self, inner, resident_mask: mx.array):
        self.inner = inner
        self.mask = resident_mask.reshape(1, -1)

    def __call__(self, x):
        logits = self.inner(x)
        return mx.where(self.mask, logits, mx.array(NEG_INF, dtype=logits.dtype))


class _RemapSwitchGLU:
    """Map GLOBAL expert ids (from the router) to LOCAL ids of the sliced bank."""

    def __init__(self, inner, global_to_local: mx.array):
        self.inner = inner
        self.g2l = global_to_local

    def __call__(self, x, indices, *args, **kwargs):
        return self.inner(x, self.g2l[indices], *args, **kwargs)


_MOE_BLOCK_ATTRS = ("mlp", "ffn", "block_sparse_moe")  # OLMoE/Qwen3 / V4 / Mixtral


def _moe_blocks(model):
    blocks = []
    for layer in model.model.layers:
        for attr in _MOE_BLOCK_ATTRS:
            b = getattr(layer, attr, None)
            if b is not None and hasattr(b, "switch_mlp"):
                blocks.append(b)
                break
    return blocks


def _slice_experts_inplace(switch_mlp, rids: mx.array, n_experts: int):
    """Fancy-index every per-expert array (axis 0) down to the resident ids.

    Materializes each sliced proj immediately (eval + free source) so we never
    hold all full experts at once: the lazy gather over the full [E,...] array
    transiently materializes that one proj, then it's freed, so the transient
    peak is ~one full proj rather than the whole expert set. Without this, the
    full set materializes during a deferred eval and the saving is lost.
    """
    for proj in (switch_mlp.gate_proj, switch_mlp.up_proj, switch_mlp.down_proj):
        for attr in ("weight", "scales", "biases", "bias"):
            v = getattr(proj, attr, None)
            if isinstance(v, mx.array) and v.shape[0] == n_experts:
                sliced = v[rids]
                mx.eval(sliced)          # page in only resident rows; drop full source
                setattr(proj, attr, sliced)


_LINEAR_LIKE = (nn.Linear, getattr(nn, "QuantizedLinear", nn.Linear))


def _mask_gate(block, mask: mx.array):
    """Restrict a block's router to the resident set, idempotently.

    OLMoE: gate is a (quantized) Linear returning logits -> wrap with _MaskedGate.
    V4: gate is a MoEGate -> set its opt-in _resident_mask (masks scores before
    argpartition). Hash-routed MoEGate layers are never passed here (they keep
    full routing).
    """
    g = block.gate
    if isinstance(g, _MaskedGate):
        return  # already masked (e.g. check mode re-installs)
    if isinstance(g, _LINEAR_LIKE):
        block.gate = _MaskedGate(g, mask)
    elif hasattr(g, "_resident_mask"):
        g._resident_mask = mask
    else:
        raise SystemExit(f"unmaskable gate type {type(g).__name__}")


def _prunable(i, rids_list, n_experts, hash_layers):
    """A layer is pruned only if it's score-routed and the bank is a real subset."""
    return i not in hash_layers and len(rids_list) < n_experts


def install_resident(model, manifest):
    """Slice each score-routed MoE block to its resident bank + arm mask/remap.
    Hash-routed layers (manifest['hash_layers']) keep full routing/weights."""
    E = manifest["n_experts"]
    blocks = _moe_blocks(model)
    per_layer = manifest["per_layer_resident"]
    hash_layers = set(manifest.get("hash_layers", []))
    if len(blocks) != len(per_layer):
        raise SystemExit(f"layer mismatch: {len(blocks)} blocks vs {len(per_layer)} manifest rows")
    for i, (b, rids_list) in enumerate(zip(blocks, per_layer)):
        if not _prunable(i, rids_list, E, hash_layers):
            continue
        rids = mx.array(rids_list)
        _slice_experts_inplace(b.switch_mlp, rids, E)
        mask = np.zeros(E, dtype=bool); mask[rids_list] = True
        _mask_gate(b, mx.array(mask))
        g2l = np.zeros(E, dtype=np.int32)
        for j, g in enumerate(rids_list):
            g2l[g] = j
        b.switch_mlp = _RemapSwitchGLU(b.switch_mlp, mx.array(g2l))


def install_mask_only(model, manifest):
    """Full switch_mlp, routing masked to the resident set (the behavior
    reference the resident-loaded model must match). Score layers only."""
    E = manifest["n_experts"]
    hash_layers = set(manifest.get("hash_layers", []))
    for i, (b, rids_list) in enumerate(zip(_moe_blocks(model), manifest["per_layer_resident"])):
        if not _prunable(i, rids_list, E, hash_layers):
            continue
        mask = np.zeros(E, dtype=bool); mask[rids_list] = True
        _mask_gate(b, mx.array(mask))


def ttft_seconds(model, tokenizer, prompt="The capital of France is") -> float:
    ids = mx.array([tokenizer.encode(prompt)])
    t0 = time.time()
    logits = model(ids)
    mx.eval(logits[:, -1, :])  # first-token logits = time-to-first-token proxy
    return time.time() - t0


def measure(model, tokenizer, label, load_secs):
    # active = steady-state resident footprint (what a long-running server holds);
    # peak = max transient (a lazy-gather slice transiently materializes the full
    # source array, so peak reflects cold-load, active reflects steady state).
    active = mx.get_active_memory() / 1e9
    tt = ttft_seconds(model, tokenizer)
    ppl = {d: perplexity(model, tokenizer, t) for d, t in EVAL_TEXT.items()}
    peak = mx.get_peak_memory() / 1e9
    print(f"[{label}] active_mem={active:.2f}GB peak_mem={peak:.2f}GB  "
          f"load+eval={load_secs:.2f}s(warm)  ttft={tt*1000:.0f}ms  ppl="
          + " ".join(f"{d}:{ppl[d]:.2f}" for d in ppl))
    return {"label": label, "active_mem_gb": active, "peak_mem_gb": peak,
            "load_eval_s": load_secs, "ttft_ms": tt * 1000, "ppl": ppl}


def _self_test_v4() -> None:
    """Synthetic masked-vs-resident check on a small score-routed DeepseekV4MoE
    (no model download): proves the V4 MoEGate score-masking + slice + global->
    local remap is behavior-preserving before running on the 149 GiB model."""
    from types import SimpleNamespace
    from mlx_lm.models.deepseek_v4 import DeepseekV4MoE
    mx.random.seed(0)
    E, k, H, I = 32, 4, 32, 64
    args = SimpleNamespace(
        n_routed_experts=E, num_experts_per_tok=k, hidden_size=H, moe_intermediate_size=I,
        n_shared_experts=1, num_hash_layers=3, scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5, norm_topk_prob=True, vocab_size=64)
    block = DeepseekV4MoE(args, layer_idx=5)  # >= num_hash_layers -> score-routed
    mx.eval(block.parameters())
    block.gate.weight = mx.random.normal((E, H)); block.gate._weight_t = None
    mx.eval(block.parameters())
    x = mx.random.normal((1, 12, H)); ids = mx.random.randint(0, 64, (1, 12))

    rids_list = list(range(0, E, 2))  # 16 distinct resident experts (>= top_k)
    mask = np.zeros(E, dtype=bool); mask[rids_list] = True

    block.gate._resident_mask = mx.array(mask)        # masked-full reference
    out_masked = block(x, ids); mx.eval(out_masked)

    _slice_experts_inplace(block.switch_mlp, mx.array(rids_list), E)  # unload to resident
    g2l = np.zeros(E, dtype=np.int32)
    for j, g in enumerate(rids_list):
        g2l[g] = j
    block.switch_mlp = _RemapSwitchGLU(block.switch_mlp, mx.array(g2l))
    out_res = block(x, ids); mx.eval(out_res)

    diff = float(mx.abs(out_masked - out_res).max().item())
    scale = float(mx.abs(out_masked).max().item())
    print(f"[v4-selftest] score-layer masked-vs-resident max|diff|={diff:.3e} rel={diff/scale:.2e}")
    if diff / scale >= 5e-3:
        raise SystemExit("v4 self-test FAILED")
    print("[v4-selftest] PASS (V4 mask+slice+remap is behavior-preserving)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/OLMoE-1B-7B-0125-Instruct-4bit")
    ap.add_argument("--mode", choices=["full", "resident", "check", "self-test-v4",
                                       "switch", "tps"], required=True)
    ap.add_argument("--manifest", help="resident-manifest/1 JSON (resident/check/switch/tps)")
    ap.add_argument("--switch-to", help="second bank manifest (switch mode)")
    ap.add_argument("--gen-tokens", type=int, default=64, help="tokens to generate (tps mode)")
    ap.add_argument("--rel-tol", type=float, default=5e-3)
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.mode == "self-test-v4":
        _self_test_v4()
        return
    from mlx_lm import load

    mx.reset_peak_memory()

    if args.mode == "full":
        t0 = time.time()
        model, tok = load(args.model, lazy=True)
        mx.eval(model.parameters())
        rec = measure(model, tok, "full", time.time() - t0)
        if args.out:
            json.dump(rec, open(args.out, "w"), indent=2)
        return

    if args.mode in ("resident", "check", "switch") and not args.manifest:
        raise SystemExit(f"--manifest required for {args.mode}")
    manifest = json.load(open(args.manifest)) if args.manifest else None

    if args.mode == "resident":
        t0 = time.time()
        model, tok = load(args.model, lazy=True)
        install_resident(model, manifest)
        mx.eval(model.parameters())  # only resident rows page in
        rec = measure(model, tok, f"resident[{manifest['bank_name']}]", time.time() - t0)
        rec["manifest"] = args.manifest
        if args.out:
            json.dump(rec, open(args.out, "w"), indent=2)
        return

    if args.mode == "switch":
        # Cost of switching the resident bank when the workload changes. We
        # measure the COLD re-materialize of a bank (load lazy -> install ->
        # eval) and the inter-bank Jaccard; a warm incremental switch (keeping
        # the full model mmap'd) re-materializes only the (1-Jaccard) of experts
        # the new bank adds.
        if not args.switch_to:
            raise SystemExit("--switch-to (second bank) required for switch mode")
        man_b = json.load(open(args.switch_to))
        t0 = time.time()
        model, tok = load(args.model, lazy=True)
        install_resident(model, man_b)
        mx.eval(model.parameters())
        t_materialize = time.time() - t0
        a, b = manifest["per_layer_resident"], man_b["per_layer_resident"]
        jac = []
        for ra, rb in zip(a, b):
            sa, sb = set(ra), set(rb)
            u = len(sa | sb)
            jac.append(len(sa & sb) / u if u else 1.0)
        J = sum(jac) / len(jac)
        peak = mx.get_peak_memory() / 1e9
        print(f"[switch {manifest['bank_name']}->{man_b['bank_name']}] "
              f"cold re-materialize={t_materialize:.2f}s peak={peak:.2f}GB  "
              f"Jaccard(A,B)={J:.2f}  -> warm incremental switch re-materializes "
              f"~{100*(1-J):.0f}% of a bank (~{t_materialize*(1-J):.2f}s est.)")
        if args.out:
            json.dump({"from": manifest["bank_name"], "to": man_b["bank_name"],
                       "cold_rematerialize_s": t_materialize, "jaccard": J,
                       "incremental_frac": 1 - J, "peak_gb": peak}, open(args.out, "w"), indent=2)
        return

    if args.mode == "tps":
        # Batch-1 decode throughput: does residency change tok/s? Prior: ~neutral
        # (only top-k experts compute per token regardless of resident count).
        from mlx_lm import generate
        model, tok = load(args.model, lazy=True)
        label = "full"
        if args.manifest:
            install_resident(model, manifest)
            label = f"resident[{manifest['bank_name']}]"
        mx.eval(model.parameters())
        prompt = "Write a short essay about the history of the printing press."
        # warmup (compile/caches), then timed run
        generate(model, tok, prompt=prompt, max_tokens=8)
        t0 = time.time()
        generate(model, tok, prompt=prompt, max_tokens=args.gen_tokens)
        dt = time.time() - t0
        tps = args.gen_tokens / dt
        peak = mx.get_peak_memory() / 1e9
        print(f"[tps {label}] {tps:.1f} tok/s ({args.gen_tokens} tok in {dt:.2f}s, batch=1) "
              f"peak={peak:.2f}GB")
        if args.out:
            json.dump({"label": label, "tps": tps, "gen_tokens": args.gen_tokens,
                       "peak_gb": peak}, open(args.out, "w"), indent=2)
        return

    # check: masked-full (behavior reference) vs resident (unloaded) on one prompt.
    model, tok = load(args.model, lazy=True)
    mx.eval(model.parameters())
    ids = mx.array([tok.encode("The capital of France is")])

    install_mask_only(model, manifest)
    masked_logits = model(ids)[:, -1, :]
    mx.eval(masked_logits)
    masked_ppl = {d: perplexity(model, tok, t) for d, t in EVAL_TEXT.items()}

    # Now actually unload to the resident bank (in place) and recompute.
    install_resident(model, manifest)  # re-wraps gate (idempotent mask) + slices + remaps
    res_logits = model(ids)[:, -1, :]
    mx.eval(res_logits)
    res_ppl = {d: perplexity(model, tok, t) for d, t in EVAL_TEXT.items()}

    diff = float(mx.abs(masked_logits - res_logits).max().item())
    scale = float(mx.abs(masked_logits).max().item())
    rel = diff / scale
    top1 = float((masked_logits.argmax(-1) == res_logits.argmax(-1)).astype(mx.float32).mean().item())
    ok = (top1 == 1.0) and (rel < args.rel_tol)
    print(f"[check {manifest['bank_name']}] masked-vs-resident: top1_match={top1:.3f} "
          f"rel={rel:.2e} (tol {args.rel_tol:.0e}) -> {'PASS' if ok else 'FAIL'}")
    print(f"  masked  ppl: " + " ".join(f"{d}:{masked_ppl[d]:.2f}" for d in masked_ppl))
    print(f"  resident ppl: " + " ".join(f"{d}:{res_ppl[d]:.2f}" for d in res_ppl))
    if args.out:
        json.dump({"bank": manifest["bank_name"], "top1_match": top1, "rel_diff": rel,
                   "passed": ok, "masked_ppl": masked_ppl, "resident_ppl": res_ppl},
                  open(args.out, "w"), indent=2)
    if not ok:
        raise SystemExit("CHECK FAILED: resident load is not behavior-preserving")


if __name__ == "__main__":
    main()
