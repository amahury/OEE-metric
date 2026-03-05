# plot_phase3_grn_benchmark.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("Figures", exist_ok=True)

# If your CSV lives in phase3_results/, change this path accordingly
df = pd.read_csv("phase3_results/grn_benchmark.csv")

g = (df.groupby("model_path")
       .agg(N=("N","first"),
            T=("T","first"),
            omega_mean=("omega","mean"),
            omega_std=("omega","std"),
            omega_canon_mean=("omega_canon","mean"),
            omega_canon_std=("omega_canon","std"))
       .reset_index())

x = np.arange(len(g))
w = 0.35

plt.figure(figsize=(7, 3))
plt.bar(x - w/2, g["omega_mean"], yerr=g["omega_std"], width=w, capsize=3, label=r"$\Omega$")
plt.bar(x + w/2, g["omega_canon_mean"], yerr=g["omega_canon_std"], width=w, capsize=3, label=r"$\Omega_{\mathrm{canon}}$")

labels = [f"{m}\nN={n}" for m, n in zip(g["model_path"], g["N"])]
plt.xticks(x, labels)
plt.ylabel(r"$\Omega(T)$")
plt.title(f"Curated Boolean GRN benchmark (T={int(g['T'].iloc[0])})")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("Figures/S24_grn_benchmark.pdf")
print("[DONE] wrote Figures/S24_grn_benchmark.pdf")
