#!/usr/bin/env python3
"""Render the WCER concentration-vs-savings chart from RESULTS.md.

This keeps the headline chart reproducible from the public benchmark table.
Outputs:
  - concentration-vs-savings.png
  - concentration-vs-savings.svg
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "RESULTS.md"
OUT_PNG = Path(__file__).with_name("concentration-vs-savings.png")
OUT_SVG = Path(__file__).with_name("concentration-vs-savings.svg")


def parse_results(md: str) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    in_table = False
    for raw in md.splitlines():
        line = raw.strip()
        if line.startswith("| model | family | imbalance"):
            in_table = True
            continue
        if in_table and not line.startswith("|"):
            break
        if not in_table or line.startswith("|---"):
            continue
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) != 6:
            continue
        model, family, imbalance, cov90, usable, ram_cut = parts
        if model == "model":
            continue
        rows.append(
            {
                "model": model,
                "family": family,
                "imbalance": float(re.sub(r"[~×x ]", "", imbalance)),
                "cov90": float(re.sub(r"[^0-9.]", "", cov90)),
                "ram_cut": float(re.sub(r"[^0-9.]", "", ram_cut)),
            }
        )
    if not rows:
        raise SystemExit("Could not parse concentration table from RESULTS.md")
    return rows


def main() -> None:
    data = parse_results(RESULTS.read_text())
    data.sort(key=lambda row: row["cov90"], reverse=True)

    x = [row["cov90"] for row in data]
    y = [row["ram_cut"] for row in data]
    labels = [row["model"].replace("-4bit", "").replace("-0125-Instruct", "") for row in data]
    label_offsets = {
        "Mixtral-8x7B-Instruct-v0.1": (10, 7),
        "OLMoE-1B-7B": (10, -18),
        "DeepSeek-V2-Lite-Chat": (10, 13),
        "Qwen3-30B-A3B-mixed-3": (10, -10),
        "DeepSeek-V4-Flash": (10, 7),
    }

    fig, ax = plt.subplots(figsize=(11.5, 7.5), dpi=220)
    fig.patch.set_facecolor("#081120")
    ax.set_facecolor("#0b1626")

    ax.scatter(x, y, s=170, color="#f5b942", edgecolors="#ffffff", linewidths=1.2, zorder=3)

    for xi, yi, label in zip(x, y, labels):
        xytext = label_offsets.get(label, (10, 7))
        ax.annotate(
            label,
            (xi, yi),
            textcoords="offset points",
            xytext=xytext,
            ha="left",
            fontsize=10,
            color="#dce6f2",
            weight="bold",
        )

    # Fit a simple line for visual guidance.
    m, b = [float(v) for v in np.polyfit(x, y, 1)]
    xs = np.linspace(min(x) - 2, max(x) + 2, 200)
    ax.plot(xs, m * xs + b, color="#5cc8ff", linewidth=2.4, alpha=0.9, zorder=2)

    ax.set_title("WCER payoff is predictable from routing concentration", fontsize=20, color="white", pad=18, weight="bold")
    ax.text(
        0.5,
        1.01,
        "x = experts needed for 90% of routing   |   y = memory cut at the best behavior-preserving resident set",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=11,
        color="#9cc0df",
    )

    ax.set_xlim(20, 92)
    ax.set_ylim(0, 74)
    ax.set_xlabel("Experts needed for 90% of routing (%)", fontsize=12, color="#dce6f2", labelpad=10)
    ax.set_ylabel("Memory cut at full quality (%)", fontsize=12, color="#dce6f2", labelpad=10)
    ax.tick_params(colors="#c8d4e3", labelsize=10)
    for spine in ax.spines.values():
        spine.set_color("#5b6f86")
    ax.grid(True, color="#203044", linewidth=0.8, alpha=0.55)

    ax.annotate(
        "More concentrated routing\n→ more memory saved",
        xy=(38, 47),
        xytext=(52, 58),
        arrowprops=dict(arrowstyle="->", color="#8de3a6", lw=2),
        fontsize=11,
        color="#bfeecf",
        ha="left",
        va="center",
        bbox=dict(boxstyle="round,pad=0.4", fc="#0f2330", ec="#3c8d5b", alpha=0.95),
    )

    ax.annotate(
        "Measured from one trace\nbefore deploying",
        xy=(66, 23),
        xytext=(59, 9),
        arrowprops=dict(arrowstyle="->", color="#f5b942", lw=1.8),
        fontsize=10,
        color="#ffe2a5",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.4", fc="#1b1a14", ec="#8d6a1a", alpha=0.95),
    )

    fig.text(
        0.5,
        0.02,
        "WCER is concentration-gated: trace first, then decide whether residency is worth it.",
        ha="center",
        va="bottom",
        fontsize=11,
        color="#8fb7d9",
    )

    fig.tight_layout(rect=(0.02, 0.05, 0.98, 0.94))
    fig.savefig(OUT_PNG, facecolor=fig.get_facecolor(), bbox_inches="tight")
    fig.savefig(OUT_SVG, facecolor=fig.get_facecolor(), bbox_inches="tight")


if __name__ == "__main__":
    main()
