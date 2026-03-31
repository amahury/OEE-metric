#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def pretty_label(row):
    model = row["model_path"]
    if row["family"] == "GRN":
        if model == "bcell_diff.bnet":
            return "B-cell diff.\nN=22"
        if model == "cardi_devel.bnet":
            return "Cardiac devel.\nN=15"
        if model == "tcell_diff.bnet":
            return "T-cell diff.\nN=23"
        return f"{model}\nN={int(row['N'])}"

    # RBN refs
    if "ordered" in model:
        return f"RBN ordered\nN={int(row['N'])}, K={row['K_ref']}"
    if "critical" in model:
        return f"RBN near-critical\nN={int(row['N'])}, K={row['K_ref']}"
    if "chaotic" in model:
        return f"RBN chaotic\nN={int(row['N'])}, K={row['K_ref']}"
    return f"{model}\nN={int(row['N'])}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default="phase3_results/grn_benchmark.csv")
    ap.add_argument("--outfile", default="Figures/S26_grn_benchmark.pdf")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.outfile) or ".", exist_ok=True)

    df = pd.read_csv(args.infile)

    g = (
        df.groupby(["family", "model_path", "N", "T", "K_ref"], dropna=False)
          .agg(
              omega_mean=("omega", "mean"),
              omega_std=("omega", "std"),
              omega_canon_mean=("omega_canon", "mean"),
              omega_canon_std=("omega_canon", "std"),
          )
          .reset_index()
    )

    # Order: curated GRNs first, then RBN refs
    desired_order = [
        "bcell_diff.bnet",
        "cardi_devel.bnet",
        "tcell_diff.bnet",
        "RBN_ordered_K1.5",
        "RBN_critical_K2.1",
        "RBN_chaotic_K3.5",
    ]
    g["order"] = g["model_path"].map({m: i for i, m in enumerate(desired_order)})
    g = g.sort_values(["order", "family"]).reset_index(drop=True)

    x = np.arange(len(g))
    colors = ["tab:blue" if fam == "GRN" else "tab:orange" for fam in g["family"]]

    plt.figure(figsize=(10, 4.8))
    plt.bar(
        x,
        g["omega_mean"],
        yerr=g["omega_std"],
        capsize=3,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )

    T = int(g["T"].iloc[0])
    plt.axhline(1.0 / T, linestyle="--", linewidth=1, label=r"$1/T$ reference")

    plt.yscale("log")
    plt.ylabel(r"$\Omega(T)$")
    plt.xticks(x, [pretty_label(row) for _, row in g.iterrows()])
    plt.title(f"Curated Boolean GRN benchmark with size-matched RBN references (T={T})")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(args.outfile)
    print(f"[DONE] wrote {args.outfile}")

if __name__ == "__main__":
    main()