#!/usr/bin/env python3
import os
import csv
import re
import argparse
import numpy as np

from single import extract_attractor_metrics


# -------------------- helpers --------------------
def _safe_base(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_]", "_", name)

def _make_unique_name_map(names):
    """
    Map raw names -> safe unique python identifiers.
    If collisions occur after sanitization, suffix with _2, _3, ...
    """
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
    """
    Extract candidate node tokens from RHS in BoolNet/CellCollective-like syntax.
    Includes names like IL-4, exogen_BMP2_II, NF_kB, etc.
    """
    toks = re.findall(r"[A-Za-z_][A-Za-z0-9_\-]*", expr)
    bad = {"and", "or", "not", "True", "False"}
    return [t for t in toks if t not in bad]


# -------------------- .bnet parser --------------------
def load_bnet(path):
    """
    BoolNet-style lines:
        Node, RHS
        Node <- RHS

    Returns:
        nodes_safe: list[str]  (safe identifiers)
        compiled_rules: dict[node_safe] = compiled python code object
        name_map: dict[raw_name] = safe_name
    """
    raw_nodes = []
    raw_exprs = {}

    # 1) read LHS rules
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

    # 2) add missing RHS tokens as “input nodes” with self-latch rule
    seen = set(raw_nodes)
    for rhs in list(raw_exprs.values()):
        for tok in _rhs_tokens(rhs):
            if tok not in seen:
                seen.add(tok)
                raw_nodes.append(tok)
                raw_exprs[tok] = tok  # self-latch input

    # 3) build safe unique identifiers
    name_map = _make_unique_name_map(raw_nodes)

    # 4) compile rules
    compiled = {}
    for raw_n, raw_expr in raw_exprs.items():
        e = raw_expr

        # normalize operators
        e = e.replace("!", " not ")
        e = e.replace("~", " not ")
        e = e.replace("&", " and ")
        e = e.replace("*", " and ")
        e = e.replace("|", " or ")
        e = e.replace("+", " or ")

        # normalize constants
        e = re.sub(r"\b0\b", "False", e)
        e = re.sub(r"\b1\b", "True", e)

        # replace node tokens by safe ids (longer names first to avoid IL-21 vs IL-2 issues)
        for old in sorted(name_map.keys(), key=len, reverse=True):
            e = re.sub(rf"\b{re.escape(old)}\b", name_map[old], e)

        node_safe = name_map[raw_n]
        compiled[node_safe] = compile(e, f"<bnet:{os.path.basename(path)}:{node_safe}>", "eval")

    nodes_safe = [name_map[n] for n in raw_nodes]
    return nodes_safe, compiled, name_map


# -------------------- simulator --------------------
def simulate_grn(nodes, compiled_rules, T=50000, seed=0):
    rng = np.random.default_rng(seed)
    N = len(nodes)

    # init random boolean state (incl. “inputs” we auto-added)
    x = {n: bool(rng.integers(0, 2)) for n in nodes}

    states = np.zeros((T, N), dtype=np.uint8)
    for t in range(T):
        row = np.fromiter((int(x[n]) for n in nodes), dtype=np.uint8, count=N)
        states[t] = row

        env = dict(x)  # locals for eval
        x_next = {}
        for n in nodes:
            x_next[n] = bool(eval(compiled_rules[n], {"__builtins__": {}}, env))

        # fast stop if fixed point (very common in curated GRNs)
        if x_next == x and t < T - 1:
            states[t + 1 :] = row
            break

        x = x_next

    return states


# -------------------- benchmark runner --------------------
def run_grn_benchmark(
    bnet_path,
    out_csv="phase3_results/grn_benchmark.csv",
    T=200000,
    seeds=(0, 1, 2, 3, 4),
    resume=True,
):
    out_dir = os.path.dirname(out_csv) or "."
    os.makedirs(out_dir, exist_ok=True)

    model_base = os.path.basename(bnet_path)

    # Resume must be keyed by (model, seed, T) — not just seed
    done = set()
    if resume and os.path.exists(out_csv):
        with open(out_csv, "r", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                done.add((row["model_path"], int(row["seed"]), int(row["T"])))

    nodes, rules, _name_map = load_bnet(bnet_path)
    N = len(nodes)

    newfile = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["model_path", "seed", "N", "T", "omega", "omega_canon"],
        )
        if newfile:
            w.writeheader()

        for seed in seeds:
            key = (model_base, int(seed), int(T))
            if resume and key in done:
                continue

            states = simulate_grn(nodes, rules, T=T, seed=seed)
            _, _, KD = extract_attractor_metrics(states, canonicalize_complement=False)
            _, _, KDc = extract_attractor_metrics(states, canonicalize_complement=True)

            w.writerow(
                {
                    "model_path": model_base,
                    "seed": int(seed),
                    "N": int(N),
                    "T": int(T),
                    "omega": float(KD / (T**2)),
                    "omega_canon": float(KDc / (T**2)),
                }
            )
            f.flush()


def _parse_seeds(s):
    return tuple(int(x) for x in s.split(",") if x.strip() != "")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("bnet_path")
    ap.add_argument("--out", default="phase3_results/grn_benchmark.csv")
    ap.add_argument("--T", type=int, default=200000)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    run_grn_benchmark(
        args.bnet_path,
        out_csv=args.out,
        T=args.T,
        seeds=_parse_seeds(args.seeds),
        resume=(not args.no_resume),
    )