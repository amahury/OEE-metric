# Parameters & Environment Variables

## Boolean Network
- `N` (default: 100 nodes)
- `K` grid: `[1.1, 1.3, ..., 4.5]`

## Horizon
- `T` (e.g. `1e6`), override with `OEE_STEPS`.

## GPU / Streaming
- `OEE_USE_GPU`, `OEE_GPU_MIN_T`, `OEE_GPU_UPDATE` (`single.py`)
- `OEE_USE_GPU`, `OEE_GPU_MIN_T`, `OEE_UPDATE` (`parallel_new.py`)
- `OEE_GPU_CHUNK_T`, `OEE_GPU_TPB_X`, `OEE_GPU_TPB_Y`

## PBN & Sampling
- `OEE_PBN_DEPENDENT`
- `OEE_CI_DELTA`, `OEE_CI_BATCH`, `OEE_MAX_NETS`

## LUT size cap
- `OEE_MAX_K` (cap on in-degree for tractable LUTs)
