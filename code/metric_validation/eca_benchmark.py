import os
import csv
import zlib
import numpy as np

from single import extract_attractor_metrics

# ---- ECA core -------------------------------------------------------------
def eca_step(state, rule_bits):
    # state: (N,) uint8 in {0,1}
    left  = np.roll(state, 1)
    self_ = state
    right = np.roll(state, -1)
    idx = (left << 2) | (self_ << 1) | right  # 0..7
    return rule_bits[idx]

def simulate_eca(rule, N, T, seed=0, init="random"):
    rng = np.random.default_rng(seed)

    # Wolfram code: bit i corresponds to neighborhood i (0..7) where i = (l<<2)|(c<<1)|r
    # In Wolfram convention, rule bits are read from 111..000; this matches indexing by i if we reverse.
    bits = np.array([(rule >> i) & 1 for i in range(8)], dtype=np.uint8)  # i=0..7 => 000..111
    # but our idx uses 0..7 for 000..111, so this is consistent.

    if init == "random":
        x = rng.integers(0, 2, size=N, dtype=np.uint8)
    elif init == "single_one":
        x = np.zeros(N, dtype=np.uint8); x[N//2] = 1
    else:
        raise ValueError("init must be 'random' or 'single_one'")

    states = np.zeros((T, N), dtype=np.uint8)
    for t in range(T):
        states[t] = x
        x = eca_step(x, bits)
    return states

# ---- Metrics --------------------------------------------------------------
def omega_from_states(states, canonical=False):
    V, P, KD = extract_attractor_metrics(states, canonicalize_complement=canonical)
    T = states.shape[0]
    return KD / (T**2)

def run_eca_benchmark(
    out_csv="phase3_results/eca_benchmark.csv",
    rules=(0, 4, 30, 54, 90, 110, 150, 184, 255),
    N=100,
    T=200000,
    seeds=(0,1,2,3,4),
    init="random",
    resume=True,
):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    done = set()
    if resume and os.path.exists(out_csv):
        with open(out_csv, "r", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                done.add((int(row["rule"]), int(row["seed"]), row["init"]))

    newfile = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "rule","seed","init","N","T",
            "omega","omega_canon"
        ])
        if newfile:
            w.writeheader()

        for rule in rules:
            for seed in seeds:
                key = (rule, seed, init)
                if resume and key in done:
                    continue

                states = simulate_eca(rule=rule, N=N, T=T, seed=seed, init=init)
                om  = omega_from_states(states, canonical=False)
                omc = omega_from_states(states, canonical=True)

                w.writerow({
                    "rule": rule, "seed": seed, "init": init,
                    "N": N, "T": T,
                    "omega": om, "omega_canon": omc
                })
                f.flush()

if __name__ == "__main__":
    # quick default
    run_eca_benchmark()
