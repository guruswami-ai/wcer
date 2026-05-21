#!/usr/bin/env python3
"""WCER Stage 3 — build a resident-set manifest ("bank") from an expert-trace.

(See docs/EXPERT_RESIDENCY_AND_PRUNING.md.)

Turns a trace's per-layer histogram into an explicit, versioned manifest: the
set of GLOBAL expert ids to keep resident per layer (the hottest `budget`).
Consumed by expert_resident_load.py to actually load/unload. Separating "decide
the resident set" (this) from "load it" (the loader) keeps the policy auditable
and lets multiple banks (general / code / math) coexist per model.

Run::
    python benchmarks/expert_resident_manifest.py --trace /tmp/olmoe-trace-general.json \
        --budget-frac 0.5 --out /tmp/olmoe-resident-general-50.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np

SCHEMA_VERSION = "resident-manifest/1"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True, help="expert-trace/1 JSON")
    ap.add_argument("--budget-frac", type=float, default=0.5,
                    help="fraction of experts kept resident per layer")
    ap.add_argument("--bank-name", help="default: <workload>-<pct>pct")
    ap.add_argument("--random", action="store_true",
                    help="ABLATION baseline: pick a RANDOM resident set per layer "
                         "(same budget) instead of the trace's hottest experts. "
                         "Isolates the value of trace-driven selection vs any subset.")
    ap.add_argument("--weighted", action="store_true",
                    help="select by router-weighted importance (per_layer_weight_mass, "
                         "needs a --capture-weights trace) instead of selection count.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    d = json.load(open(args.trace))
    hist = np.array(d["per_layer_histogram"], dtype=np.float64)  # [L, E]
    if args.weighted:
        wm = d.get("per_layer_weight_mass")
        if wm is None:
            raise SystemExit("--weighted needs a trace built with --capture-weights "
                             "(per_layer_weight_mass is null)")
        hist = np.array(wm, dtype=np.float64)  # rank by router-weighted importance
    n_layers, n_experts = hist.shape
    top_k = d["top_k"]
    hash_layers = d.get("hash_layers", [])
    # Resident count is fixed per layer; guard against a budget below top_k
    # (the router could not pick top_k resident experts -> blow-up).
    budget = max(top_k, int(round(args.budget_frac * n_experts)))

    # Score-routed layers prune to the hottest `budget`; hash-routed layers route
    # by a fixed table (not maskable), so they stay FULL (a fixed broad union).
    per_layer = []
    for l in range(n_layers):
        if l in hash_layers:
            per_layer.append(list(range(n_experts)))
        elif args.random:
            per_layer.append(sorted(int(x) for x in rng.choice(n_experts, budget, replace=False)))
        else:
            per_layer.append(sorted(int(x) for x in np.argsort(-hist[l])[:budget]))
    assert all(len(r) >= top_k for r in per_layer), "resident set < top_k in some layer"

    pct = round(100 * budget / n_experts)
    suffix = "-random" if args.random else ("-weighted" if args.weighted else "")
    bank = args.bank_name or f"{d.get('workload_label', 'set')}-{pct}pct{suffix}"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "model_id": d.get("model_id"),
        "model_revision": d.get("model_revision"),
        "n_experts": n_experts,
        "n_layers": n_layers,
        "top_k": top_k,
        "bank_name": bank,
        "built_from_trace": args.trace,
        "budget_frac": budget / n_experts,
        "n_resident_per_layer": budget,
        "hash_layers": hash_layers,  # kept full (fixed routing); pruning is score-layers-only
        "per_layer_resident": per_layer,
    }
    json.dump(manifest, open(args.out, "w"), indent=2)
    print(f"[manifest] bank='{bank}' {budget}/{n_experts} experts/layer "
          f"({pct}%), {n_layers} layers, top_k={top_k} -> {args.out}")


if __name__ == "__main__":
    main()
