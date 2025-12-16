#!/usr/bin/env bash
set -euo pipefail

# maximum number of LUTs 
export OEE_MAX_K=10

# no GUI popups; show() becomes a no-op
export MPLBACKEND=Agg
export OEE_SHOW_PLOTS=0
# also make sure we don't try to "show" from our own guard (see Option B)
export OEE_SHOW=0

# (Optional) GPU toggles you already support
# === OEE runtime knobs (safe to remove any line you don't want) ==========
export OEE_USE_GPU=1            # let parallel.py/single.py prefer GPU when possible
export OEE_GPU_MIN_T=50000      # allow T >= 50k to use GPU (your code reads this)
export OEE_GPU_UPDATE=asynchronous
export OEE_PBN_DEPENDENT=1

# Adaptive sampler precision & caps
export OEE_CI_DELTA=0.02        # 95% CI half-width target for mean OEE
export OEE_CI_BATCH=50          # sample in batches of 50 nets
export OEE_MAX_NETS=1000        # never exceed your original 1000

# Optional: lower T without editing code (keeps your defaults if unset)
#export OEE_STEPS=50000          # comment this out if you want to keep T=100000
# ========================================================================

python -u parallel_new.py
