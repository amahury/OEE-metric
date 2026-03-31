#!/usr/bin/env python3
import os
import csv
import re
import argparse
import random
import numpy as np

from single import extract_attractor_metrics
from axiomatic import generate_boolean_network, simulate_pbn


# -------------------- helpers --------------------
def _safe_base(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_]", "_", name)

def _make_unique_name_map(names):
    used = set()
    name_map = {}
    for n in names:
        base = _safe_base(n)
        cand = base
        k = 2
        while cand in used:
            cand = f"{base}_{k}"
            k += 1
        used.add(cand)
        name_map[n] = cand
    return name_map

def _rhs_tokens(expr: str):
    toks = re.findall(r"[A-Za-z_][A-Za-z0-9_\-]*", expr)
    bad = {"and", "or", "not", "True", "False"}
    return [t for t in toks if t not in bad]

def _coerce01(v):
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
        return 0

    return 0

def _states_to_uint8(states, N):
    states = list(states)
    arr = np.empty((len(states), N), dtype=np.uint8)
    for t, row in enumerate(states):
        if len(row) != N:
            raise ValueError(f"State length mismatch at t={t}: got {len(row)} expected {N}")
        arr[t, :] = [_coerce01(x) for x in row]
    return arr


# -------------------- .bnet parser --------------------
def load_bnet(path):
    raw_nodes = []
    raw_exprs = {}

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "<-" in line:
                lhs, rhs = [x.strip() for x in line.split("<-", 1)]
            elif "," in line:
                lhs, rhs = [x.strip() for x in line.split(",", 1)]
            else:
                continue
            raw_nodes.append(lhs)
            raw_exprs[lhs] = rhs

    seen = set(raw_nodes)
    for rhs in list(raw_exprs.values()):
        for tok in _rhs_tokens(rhs):
            if tok not in seen:
                seen.add(tok)
                raw_nodes.append(tok)
                raw_exprs[tok] = tok  # self-latch exogenous input

    name_map = _make_unique_name_map(raw_nodes)

    compiled = {}
    for raw_n, raw_expr in raw_exprs.items():
        e = raw_expr
        e = e.replace("!", " not ")
        e = e.replace("~", " not ")
        e = e.replace("&", " and ")
        e = e.replace("*", " and ")
        e = e.replace("|", " or ")
        e = e.replace("+", " or ")

        e = re.sub(r"\b0\b", "False", e)
        e = re.sub(r"\b1\b", "True", e)

        for old in sorted(name_map.keys(), key=len, reverse=True):
            e = re.sub(rf"\b{re.escape(old)}\b", name_map[old], e)

        node_safe = name_map[raw_n]
        compiled[node_safe] = compile(e, f"<bnet:{os.path.basename(path)}:{node_safe}>", "eval")

    nodes_safe = [name_map[n] for n in raw_nodes]
    return nodes_safe, compiled


# -------------------- simulators --------------------
def simulate_grn(nodes, compiled_rules, T=50000, seed=0):
    rng = np.random.default_rng(seed)
    N = len(nodes)

    x = {n: bool(rng.integers(0, 2)) for n in nodes}

    states = np.zeros((T, N), dtype=np.uint8)
    for t in range(T):
        row = np.fromiter((int(x[n]) for n in nodes), dtype=np.uint8, count=N)
        states[t] = row

        env = dict(x)
        x_next = {}
        for n in nodes:
            x_next[n] = bool(eval(compiled_rules[n], {"__builtins__": {}}, env))

        if x_next == x and t < T - 1:
            states[t + 1 :] = row
            break

        x = x_next

    return states

def simulate_rbn_reference(N=22, K=2.1, T=200000, seed=0, topology="Poisson", bias=0.5):
    np.random.seed(seed)
    random.seed(seed)
    nodes, functions = generate_boolean_network(N, K, topology, bias)
    states = simulate_pbn([(nodes, functions, {})], [1.0], T, use_gpu=False)
    return _states_to_uint8(states, N)


# -------------------- metrics --------------------
def _omegas_from_states(states):
    _, _, KD = extract_attractor_metrics(states, canonicalize_complement=False)
    _, _, KDc = extract_attractor_metrics(states, canonicalize_complement=True)
    T = states.shape[0]
    return float(KD / (T**2)), float(KDc / (T**2))


# -------------------- benchmark runner --------------------
def run_benchmark(
    bnet_paths,
    out_csv="phase3_results/grn_benchmark.csv",
    T=200000,
    seeds=(0, 1, 2, 3, 4),
    resume=True,
    add_rbn_controls=True,
    ref_N=None,
    ref_topology="Poisson",
    ref_bias=0.5,
):
    out_dir = os.path.dirname(out_csv) or "."
    os.makedirs(out_dir, exist_ok=True)

    loaded = []
    for p in bnet_paths:
        nodes, rules = load_bnet(p)
        loaded.append((os.path.basename(p), len(nodes), nodes, rules))

    if ref_N is None:
        ref_N = int(np.median([N for _, N, _, _ in loaded]))

    reference_Ks = {
        "RBN_ordered_K1.5": 1.5,
        "RBN_critical_K2.1": 2.1,
        "RBN_chaotic_K3.5": 3.5,
    }

    done = set()
    if resume and os.path.exists(out_csv):
        with open(out_csv, "r", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                done.add((
                    row["family"],
                    row["model_path"],
                    int(row["seed"]),
                    int(row["T"]),
                    row["K_ref"],
                ))

    newfile = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "family", "model_path", "seed", "N", "T", "K_ref",
                "omega", "omega_canon"
            ],
        )
        if newfile:
            w.writeheader()

        # curated GRNs
        for model_base, N, nodes, rules in loaded:
            for seed in seeds:
                key = ("GRN", model_base, int(seed), int(T), "")
                if resume and key in done:
                    continue

                states = simulate_grn(nodes, rules, T=T, seed=seed)
                omega, omega_canon = _omegas_from_states(states)

                w.writerow({
                    "family": "GRN",
                    "model_path": model_base,
                    "seed": int(seed),
                    "N": int(N),
                    "T": int(T),
                    "K_ref": "",
                    "omega": omega,
                    "omega_canon": omega_canon,
                })
                f.flush()

        # size-matched RBN controls
        if add_rbn_controls:
            for label, K in reference_Ks.items():
                for seed in seeds:
                    key = ("RBN_ref", label, int(seed), int(T), str(K))
                    if resume and key in done:
                        continue

                    states = simulate_rbn_reference(
                        N=ref_N,
                        K=K,
                        T=T,
                        seed=seed,
                        topology=ref_topology,
                        bias=ref_bias,
                    )
                    omega, omega_canon = _omegas_from_states(states)

                    w.writerow({
                        "family": "RBN_ref",
                        "model_path": label,
                        "seed": int(seed),
                        "N": int(ref_N),
                        "T": int(T),
                        "K_ref": str(K),
                        "omega": omega,
                        "omega_canon": omega_canon,
                    })
                    f.flush()


def _parse_seeds(s):
    return tuple(int(x) for x in s.split(",") if x.strip() != "")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("bnet_paths", nargs="+")
    ap.add_argument("--out", default="phase3_results/grn_benchmark.csv")
    ap.add_argument("--T", type=int, default=200000)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--no-rbn-controls", action="store_true")
    ap.add_argument("--ref-N", type=int, default=None)
    ap.add_argument("--ref-topology", default="Poisson")
    ap.add_argument("--ref-bias", type=float, default=0.5)
    args = ap.parse_args()

    run_benchmark(
        bnet_paths=args.bnet_paths,
        out_csv=args.out,
        T=args.T,
        seeds=_parse_seeds(args.seeds),
        resume=(not args.no_resume),
        add_rbn_controls=(not args.no_rbn_controls),
        ref_N=args.ref_N,
        ref_topology=args.ref_topology,
        ref_bias=args.ref_bias,
    )