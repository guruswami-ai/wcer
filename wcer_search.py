#!/usr/bin/env python3
"""WCER hardware-fit search — automate the residency × precision × selection grid.

The "front end for hardware-conditioned model fitting": sweep resident budget
(and, via multiple --model precisions, quantization level) × selection criterion,
measure RAM / quality / TTFT per grid point in ISOLATED processes (clean peak
memory), and emit a Pareto table — the best quality/memory tradeoff for a machine
and workload. (See docs/EXPERT_RESIDENCY_AND_PRUNING.md.)

Each grid point shells out to the Stage-3 manifest builder + Stage-5 loader (each
writes a metrics JSON) so peak RAM is process-isolated. The quant axis is just
"pass several --model checkpoints at different precisions"; the loop is identical.

Run::
    PYTHONPATH=$MLX_LM python \
        benchmarks/wcer_search.py \
        --model mlx-community/Qwen3-30B-A3B-mixed-3-4bit \
        --trace /tmp/qw-wt-general.json \
        --budgets 0.25 0.5 0.75 --selection weighted --out /tmp/wcer-pareto.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))


def _run(argv) -> None:
    subprocess.run(argv, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _loader(model, mode, out, manifest=None) -> dict:
    argv = [sys.executable, os.path.join(HERE, "expert_resident_load.py"),
            "--model", model, "--mode", mode, "--out", out]
    if manifest:
        argv += ["--manifest", manifest]
    _run(argv)
    return json.load(open(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", required=True,
                    help="one or more checkpoints (different precisions = the quant axis)")
    ap.add_argument("--trace", required=True, help="expert-trace JSON (per model, or shared arch)")
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    ap.add_argument("--selection", choices=["count", "weighted"], default="weighted")
    ap.add_argument("--quality-tol", type=float, default=1.10,
                    help="Pareto gate: max in-domain ppl ratio vs full to count as 'acceptable'")
    ap.add_argument("--out", default="/tmp/wcer-pareto.json")
    args = ap.parse_args()

    rows = []
    with tempfile.TemporaryDirectory() as td:
        for model in args.model:
            tag = model.split("/")[-1]
            full = _loader(model, "full", os.path.join(td, "full.json"))
            full_ppl = full["ppl"]
            rows.append({"model": tag, "budget": 1.0, "selection": "—",
                         "peak_gb": full["peak_mem_gb"], "ttft_ms": full["ttft_ms"],
                         "ppl": full_ppl, "ppl_ratio": 1.0, "acceptable": True})
            for bf in sorted(args.budgets):
                man = os.path.join(td, f"man_{tag}_{bf}.json")
                margv = [sys.executable, os.path.join(HERE, "expert_resident_manifest.py"),
                         "--trace", args.trace, "--budget-frac", str(bf), "--out", man]
                if args.selection == "weighted":
                    margv.append("--weighted")
                _run(margv)
                m = _loader(model, "resident", os.path.join(td, f"res_{tag}_{bf}.json"), manifest=man)
                # in-domain = the trace's workload (first eval domain) — use 'general'
                dom = "general" if "general" in full_ppl else next(iter(full_ppl))
                ratio = m["ppl"][dom] / full_ppl[dom]
                rows.append({"model": tag, "budget": bf, "selection": args.selection,
                             "peak_gb": m["peak_mem_gb"], "ttft_ms": m["ttft_ms"],
                             "ppl": m["ppl"], "ppl_ratio": ratio,
                             "acceptable": ratio <= args.quality_tol})

    # Pareto frontier: among acceptable points, those not dominated (lower RAM AND
    # better-or-equal quality) by another acceptable point.
    acc = [r for r in rows if r["acceptable"]]
    pareto = []
    for r in acc:
        if not any(o is not r and o["peak_gb"] <= r["peak_gb"] and o["ppl_ratio"] <= r["ppl_ratio"]
                   and (o["peak_gb"] < r["peak_gb"] or o["ppl_ratio"] < r["ppl_ratio"]) for o in acc):
            pareto.append(r)
    pareto_keys = {(r["model"], r["budget"]) for r in pareto}

    json.dump({"rows": rows, "pareto": pareto, "quality_tol": args.quality_tol},
              open(args.out, "w"), indent=2)

    dom_hdr = "in-domain ppl (×full)"
    print(f"WCER hardware-fit search  (selection={args.selection}, "
          f"acceptable = in-domain ppl ≤ {args.quality_tol:.2f}×full)\n")
    print(f"  {'model':<28} {'budget':>7} {'peak_GB':>8} {'TTFT_ms':>8} {dom_hdr:>22}  {'':>6}")
    for r in rows:
        dom = "general" if "general" in r["ppl"] else next(iter(r["ppl"]))
        star = " ★PARETO" if (r["model"], r["budget"]) in pareto_keys else (
            "" if r["acceptable"] else "  (over tol)")
        b = "full" if r["budget"] == 1.0 else f"{r['budget']:.0%}"
        print(f"  {r['model']:<28} {b:>7} {r['peak_gb']:>7.1f} {r['ttft_ms']:>7.0f} "
              f"{r['ppl'][dom]:>12.2f} (×{r['ppl_ratio']:.2f}){star}")
    print(f"\nPareto-optimal (min RAM at acceptable quality): "
          + ", ".join(f"{r['model']}@{r['budget']:.0%}({r['peak_gb']:.1f}GB)" for r in pareto))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
