import os
import csv
import zlib
import numpy as np

from single import extract_attractor_metrics
from axiomatic import (
    generate_boolean_network,
    apply_modal_logic,
    apply_paraconsistent_logic,
    simulate_pbn,
)
import random

def _coerce01(v):
    """
    Convert any weird node value into a strict binary 0/1.
    - tuples like (2, x) -> x
    - bool/int/np.int -> int & 1
    - strings -> best effort (defaults to 0)
    """
    if isinstance(v, tuple) and len(v) >= 2:
        v = v[1]
    elif isinstance(v, tuple) and len(v) == 1:
        v = v[0]

    if isinstance(v, (np.integer, int, bool)):
        return int(v) & 1

    if isinstance(v, float):
        return int(round(v)) & 1

    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"1", "true", "t", "yes"}:
            return 1
        if s in {"0", "false", "f", "no"}:
            return 0
        # unknown markers like "possible"/"superposed" -> 0
        return 0

    # fallback
    return 0


def _states_to_uint8(states, N):
    """
    Convert list/generator of states into (T,N) uint8 array.
    Ensures every row has length N and values are 0/1.
    """
    states = list(states)
    arr = np.empty((len(states), N), dtype=np.uint8)
    for t, row in enumerate(states):
        if len(row) != N:
            raise ValueError(f"State length mismatch at t={t}: got {len(row)} expected {N}")
        arr[t, :] = [_coerce01(x) for x in row]
    return arr

# ---- Baselines ------------------------------------------------------------
def unique_state_fraction(states):
    # states: (T,N) uint8
    packed = np.packbits(states.astype(np.uint8), axis=1)
    sset = set(packed[i].tobytes() for i in range(packed.shape[0]))
    return len(sset) / states.shape[0]

def mean_node_entropy(states):
    # mean over nodes of Bernoulli entropy h(p)
    eps = 1e-12
    p = states.mean(axis=0).clip(eps, 1-eps)
    h = -(p*np.log2(p) + (1-p)*np.log2(1-p))
    return float(h.mean())

def zlib_compression_ratio(states):
    packed = np.packbits(states.astype(np.uint8), axis=1)
    raw = packed.tobytes()
    comp = zlib.compress(raw, level=9)
    return len(comp) / max(1, len(raw))

def compute_all_metrics(states):
    T = states.shape[0]
    V,P,KD  = extract_attractor_metrics(states, canonicalize_complement=False)
    Vc,Pc,KDc = extract_attractor_metrics(states, canonicalize_complement=True)

    return dict(
        omega=KD/(T**2),
        omega_canon=KDc/(T**2),
        unique_frac=unique_state_fraction(states),
        node_entropy=mean_node_entropy(states),
        zlib_ratio=zlib_compression_ratio(states),
    )

# ---- Trajectory generators (small-N, cheap) -------------------------------
def sim_rbn_trajectory(N=30, K=2.5, topology="Poisson", bias=0.5, T=200000, seed=0):
    np.random.seed(seed)
    random.seed(seed)
    nodes, functions = generate_boolean_network(N, K, topology, bias)
    states = simulate_pbn([(nodes, functions, {})], [1.0], T, use_gpu=False)
    return _states_to_uint8(states, N)

def sim_modal_trajectory(N=30, K=2.5, topology="Poisson", bias=0.5, T=200000, seed=0,
                         accessibility_degree=1, p_possible=0.5, p_necessary=0.5):
    np.random.seed(seed)
    random.seed(seed)
    nodes, functions = generate_boolean_network(N, K, topology, bias)
    nodes, functions, ent, kripke = apply_modal_logic(
        (nodes, functions),
        accessibility_degree=accessibility_degree,
        p_possible=p_possible,
        p_necessary=p_necessary
    )
    states = simulate_pbn([(nodes, functions, ent, kripke)], [1.0], T, use_gpu=False)
    return _states_to_uint8(states, N)

def sim_paraconsistent_trajectory(N=30, K=2.5, topology="Poisson", bias=0.5, T=200000, seed=0,
                                  contradiction_prob=0.1):
    np.random.seed(seed)
    random.seed(seed)
    nodes, functions = generate_boolean_network(N, K, topology, bias)
    nodes, functions, ent = apply_paraconsistent_logic((nodes, functions), contradiction_prob)
    states = simulate_pbn([(nodes, functions, ent)], [1.0], T, use_gpu=False)
    return _states_to_uint8(states, N)

def run_metric_comparison(
    out_csv="phase3_results/metric_comparison.csv",
    seeds=range(30),
    N=30,
    T=200000,
    resume=True,
):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    done = set()
    if resume and os.path.exists(out_csv):
        with open(out_csv, "r", newline="") as f:
            for row in csv.DictReader(f):
                done.add((row["system"], int(row["seed"])))

    newfile = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as f:
        fieldnames = ["system","seed","N","T","K","topology","bias",
                      "omega","omega_canon","unique_frac","node_entropy","zlib_ratio"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if newfile:
            w.writeheader()

        for seed in seeds:
            # 1) deterministic baseline
            key = ("RBN", seed)
            if not (resume and key in done):
                states = sim_rbn_trajectory(N=N, K=2.5, topology="Poisson", bias=0.5, T=T, seed=seed)
                m = compute_all_metrics(states)
                w.writerow(dict(system="RBN", seed=seed, N=N, T=T, K=2.5, topology="Poisson", bias=0.5, **m))
                f.flush()

            # 2) modal (high-Ω regime in your homogeneous setting)
            key = ("Modal", seed)
            if not (resume and key in done):
                states = sim_modal_trajectory(N=N, K=3.5, topology="Poisson", bias=0.5, T=T, seed=seed,
                                              accessibility_degree=1, p_possible=0.5, p_necessary=0.5)
                m = compute_all_metrics(states)
                w.writerow(dict(system="Modal", seed=seed, N=N, T=T, K=3.5, topology="Poisson", bias=0.5, **m))
                f.flush()

            # 3) paraconsistent (shoulder/critical-ish regimes)
            key = ("Paraconsistent", seed)
            if not (resume and key in done):
                states = sim_paraconsistent_trajectory(N=N, K=2.3, topology="Poisson", bias=0.5, T=T, seed=seed,
                                                      contradiction_prob=0.1)
                m = compute_all_metrics(states)
                w.writerow(dict(system="Paraconsistent", seed=seed, N=N, T=T, K=2.3, topology="Poisson", bias=0.5, **m))
                f.flush()

if __name__ == "__main__":
    run_metric_comparison()
