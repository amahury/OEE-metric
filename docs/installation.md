# Installation

## Using Conda

```bash
conda env create -f environment.yml
conda activate oee-rbn
```

## Using pip

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **GPU note:** If you have a CUDA-capable GPU, install a `cupy` wheel compatible with your CUDA version and ensure `cubewalkers` is available. Otherwise, the CPU path is used automatically.
