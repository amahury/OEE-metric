# Quickstart

### 1) Baseline Ω vs K (Homogeneous)

```bash
python single.py
```

- Produces NPY files like `open_endedness_T1000000.npy` under `data/npy/`.
- Also shows a preview plot.

### 2) Parameter Sweeps (Multiple Logics)

```bash
python parallel_new.py
```

- Writes `open_endedness_*.npy` files and `open_endedness_*.png` figures into `data/` subfolders.

### 3) Figures for Paper

Open and run:

- `notebooks/plots_homo.ipynb`
- `notebooks/plots_hetero.ipynb`

They will read from `data/npy/` and write to `data/figures/`.
