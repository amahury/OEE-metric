# Reproducing the Paper

1. **Run baselines** with `single.py` to generate Ω vs K for chosen horizons `T`.
2. **Sweep parameters** with `parallel_new.py` for PBN, Modal, Paraconsistent, Quantum, and Dynamic logics.
3. **Open notebooks**:
   - `plots_homo.ipynb` (homogeneous settings)
   - `plots_hetero.ipynb` (heterogeneous settings)

Place your existing `.npy` files (if already computed) under `data/npy/` to skip heavy runs.
