"""Plots of serving results written by run.py.

    python benchmarks/serving/plot.py results/serving/arxiv results/serving/gsm8k --out serving.pdf

One row per result directory: average TTFT, TBT and JCT per method at the first arrival rate, and the
average JCT of every method against the arrival rate.
"""

from __future__ import annotations

import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(args.dirs), 4, figsize=(16, 3 * len(args.dirs)), squeeze=False)
    for row, d in zip(axes, args.dirs):
        with open(os.path.join(d, "summary.json")) as f:
            summary = json.load(f)
        methods = list(summary)
        rates = sorted({r for m in methods for r in summary[m]}, key=float)
        first = rates[0]
        for ax, metric, label in zip(row[:3], ("avg_ttft_s", "avg_tbt_s", "avg_jct_s"), ("TTFT (s)", "TBT (s)", "JCT (s)")):
            vals = [summary[m].get(first, {}).get(metric, 0.0) for m in methods]
            ax.bar(range(len(methods)), vals, color=[f"C{i}" for i in range(len(methods))])
            ax.set_xticks(range(len(methods)), methods, rotation=30, ha="right", fontsize=8)
            ax.set_ylabel(label)
            ax.set_title(f"{os.path.basename(d.rstrip('/'))}, {first} req/s", fontsize=9)
        for i, m in enumerate(methods):
            xs = [float(r) for r in rates if r in summary[m] and r != "inf"]
            ys = [summary[m][r]["avg_jct_s"] for r in rates if r in summary[m] and r != "inf"]
            row[3].plot(xs, ys, marker="o", ms=3, color=f"C{i}", label=m)
        row[3].set_xlabel("arrival rate (req/s)")
        row[3].set_ylabel("JCT (s)")
        row[3].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
