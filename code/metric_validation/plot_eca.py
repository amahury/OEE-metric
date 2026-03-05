#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    infile = "phase3_results/eca_benchmark.csv"
    outpdf = "Figures/S23_eca_benchmark.pdf"
    os.makedirs(os.path.dirname(outpdf), exist_ok=True)

    df = pd.read_csv(infile)
    g = df.groupby("rule").agg(
        omega_mean=("omega","mean"),
        omega_std=("omega","std"),
        omega_canon_mean=("omega_canon","mean"),
        omega_canon_std=("omega_canon","std"),
    ).reset_index().sort_values("rule")

    x = np.arange(len(g))
    width = 0.38

    plt.figure(figsize=(10,4.5))
    plt.bar(x - width/2, g["omega_mean"], width, yerr=g["omega_std"], capsize=3, label="Ω")
    plt.bar(x + width/2, g["omega_canon_mean"], width, yerr=g["omega_canon_std"], capsize=3, label="Ω_canon")

    plt.xticks(x, g["rule"].astype(str).tolist())
    plt.yscale("log")
    plt.xlabel("ECA rule")
    plt.ylabel("Ω(T) (log scale)")
    plt.title("ECA benchmark + complement-canonicalization control")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpdf)
    print(f"[DONE] wrote {outpdf}")

if __name__ == "__main__":
    main()
