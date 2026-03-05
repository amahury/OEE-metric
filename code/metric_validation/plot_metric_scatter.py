#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    infile = "phase3_results/metric_comparison.csv"
    outpdf = "Figures/S25_metric_scatter.pdf"
    os.makedirs(os.path.dirname(outpdf), exist_ok=True)

    df = pd.read_csv(infile)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    for system, sub in df.groupby("system"):
        axes[0].scatter(sub["unique_frac"], sub["omega"], s=12, label=system, alpha=0.7)
        axes[1].scatter(sub["node_entropy"], sub["omega"], s=12, label=system, alpha=0.7)
        axes[2].scatter(sub["zlib_ratio"], sub["omega"], s=12, label=system, alpha=0.7)

    axes[0].set_xlabel("Unique-state fraction")
    axes[1].set_xlabel("Mean node entropy")
    axes[2].set_xlabel("Zlib ratio (compressed/raw)")

    for ax in axes:
        ax.set_ylabel("Ω(T)")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.2)

    axes[1].legend(fontsize=8, loc="best")
    fig.suptitle("Ω vs baseline trajectory indices (Phase 3)")
    plt.tight_layout()
    plt.savefig(outpdf)
    print(f"[DONE] wrote {outpdf}")

if __name__ == "__main__":
    main()
