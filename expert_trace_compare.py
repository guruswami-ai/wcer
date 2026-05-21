#!/usr/bin/env python3
"""WCER Stage 2 — cross-workload expert-usage comparison.

Tests the "Workload-Conditioned" premise of WCER: does residency need to be
workload-specific, or does one shared resident set serve everything? (See
README.md.)

Cross-workload expert-usage comparison — does residency need to be workload-specific?

Consumes >=2 ``expert-trace/1`` JSON summaries (from ``expert_trace.py``) and
answers the resident-set decision directly: if you can only keep a fraction of
each layer's experts resident, does a set tuned to workload A still serve
workload B — or do different workloads need different resident sets?

Two readouts per workload pair, averaged over layers:
  * **cross-coverage**: a resident set = the hottest ``budget`` experts of the
    ROW workload; the cell is the fraction of the COLUMN workload's routes that
    land in it. Diagonal = self-coverage (the ceiling). Big diagonal-vs-
    off-diagonal gap ⇒ specialization matters ⇒ workload-specific sets pay off.
  * **Jaccard**: overlap of the two workloads' top-``budget`` expert sets.

Run::
    python benchmarks/expert_trace_compare.py --budget-frac 0.5 \
        /tmp/olmoe-trace-general.json /tmp/olmoe-trace-code.json /tmp/olmoe-trace-math.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def load_trace(path):
    d = json.load(open(path))
    return {
        "label": d.get("workload_label", path),
        "hist": np.array(d["per_layer_histogram"], dtype=np.float64),  # [L, E]
        "n_experts": d["n_experts"],
        "n_layers": d["n_layers"],
        "hash_layers": d.get("hash_layers", []),
    }


def resident_set(hist_layer, budget):
    """Indices of the `budget` hottest experts in one layer."""
    return set(np.argsort(-hist_layer)[:budget].tolist())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+", help="expert-trace JSON files (>=2)")
    ap.add_argument("--budget-frac", type=float, default=0.5,
                    help="fraction of experts kept resident per layer")
    ap.add_argument("--score-only", action="store_true",
                    help="exclude hash-routed layers (fixed routing dilutes the signal)")
    args = ap.parse_args()
    if len(args.traces) < 2:
        raise SystemExit("need >=2 traces to compare")

    traces = [load_trace(p) for p in args.traces]
    E, L = traces[0]["n_experts"], traces[0]["n_layers"]
    for t in traces:
        if t["n_experts"] != E or t["n_layers"] != L:
            raise SystemExit("traces must be the same model (n_experts/n_layers differ)")
    budget = max(1, int(round(args.budget_frac * E)))
    labels = [t["label"] for t in traces]
    hash_layers = set(traces[0]["hash_layers"])
    layer_ids = [l for l in range(L) if not (args.score_only and l in hash_layers)]
    if args.score_only:
        print(f"(score-only: excluding {len(hash_layers)} hash layers {sorted(hash_layers)}; "
              f"comparing {len(layer_ids)} score-routed layers)\n")

    # cross_cov[a][b] = mean over layers of (column b's routes covered by row a's
    # top-budget resident set). jac[a][b] = mean Jaccard of the top-budget sets.
    n = len(traces)
    cross_cov = np.zeros((n, n))
    jac = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            covs, jacs = [], []
            for l in layer_ids:
                Ra = resident_set(traces[a]["hist"][l], budget)
                Rb = resident_set(traces[b]["hist"][l], budget)
                hb = traces[b]["hist"][l]
                tot = hb.sum()
                covered = hb[list(Ra)].sum()
                covs.append(covered / tot if tot else 0.0)
                inter = len(Ra & Rb)
                union = len(Ra | Rb)
                jacs.append(inter / union if union else 0.0)
            cross_cov[a][b] = np.mean(covs)
            jac[a][b] = np.mean(jacs)

    w = max(len(x) for x in labels) + 2
    budget_pct = 100 * budget / E
    print(f"Model: {E} experts/layer, {L} layers. "
          f"Resident budget = {budget}/{E} ({budget_pct:.0f}%) hottest experts per layer.\n")

    print(f"CROSS-COVERAGE  (row = resident set tuned to; col = workload served)")
    print(f"  cell = % of the column workload's routes covered by the row's resident set")
    print(" " * w + "".join(f"{lb:>10}" for lb in labels))
    for a in range(n):
        print(f"{labels[a]:<{w}}" + "".join(f"{100*cross_cov[a][b]:9.1f}%" for b in range(n)))

    print(f"\nJACCARD OVERLAP of top-{budget} expert sets (mean over layers)")
    print(" " * w + "".join(f"{lb:>10}" for lb in labels))
    for a in range(n):
        print(f"{labels[a]:<{w}}" + "".join(f"{jac[a][b]:9.2f} " for b in range(n)))

    # Verdict: how much coverage you lose serving a workload with the WRONG
    # (other-workload) resident set vs its own.
    diag = np.diag(cross_cov)
    off = cross_cov.copy()
    np.fill_diagonal(off, np.nan)
    worst_gap = np.nanmax(diag[:, None] - off.T)  # self minus best-foreign per col
    mean_self = diag.mean()
    mean_off = np.nanmean(off)
    print(f"\nVERDICT")
    print(f"  mean self-coverage   = {100*mean_self:.1f}%")
    print(f"  mean cross-coverage  = {100*mean_off:.1f}%")
    print(f"  max self-vs-foreign gap = {100*worst_gap:.1f} pts")
    if worst_gap > 0.05:
        print("  => workloads diverge: workload-specific resident sets recover "
              f"up to {100*worst_gap:.1f} pts of coverage a shared set would lose.")
    else:
        print("  => workloads overlap heavily: a single shared resident set suffices; "
              "workload-specific sets add little.")


if __name__ == "__main__":
    main()
