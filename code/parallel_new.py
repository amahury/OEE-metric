import os
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 1200
import time
from tqdm import tqdm
from single import extract_attractor_metrics
from axiomatic import (
    generate_boolean_network,
    apply_paraconsistent_logic,
    apply_modal_logic,
    apply_quantum_logic,
    apply_dynamic_logic,
    simulate_pbn,
    measure_kd_piecewise,  
)

import os
USE_GPU = os.environ.get("OEE_USE_GPU", "0") != "0"
GPU_MIN_T = int(os.environ.get("OEE_GPU_MIN_T", "200000"))
GPU_UPDATE = os.environ.get("OEE_UPDATE", "synchronous")   # "asynchronous", "asynchronous_set"
PBN_DEP    = os.environ.get("OEE_PBN_DEPENDENT", "1") == "1"

#New stuff
# ==== Adaptive sampling (drop-in, no disruption) =========================
def _invoke_sample_once(args):
    fn, kwargs = args
    return fn(**kwargs)

def _ci_halfwidth(samples, z=1.96):
    if len(samples) <= 1:
        return float('inf')
    s = float(np.std(samples, ddof=1))
    return z * s / np.sqrt(len(samples))

def _run_until_ci(sample_once_fn, fixed_kwargs, batch=50, max_n=1000, delta=0.02, use_pool=True):
    """
    Repeatedly call sample_once_fn(**fixed_kwargs) in small batches until
    the 95% CI half-width <= delta, or until max_n samples are drawn.
    Returns (mean, n_used).
    """
    samples = []
    while len(samples) < max_n:
        n_to_run = min(batch, max_n - len(samples))
        if use_pool and n_to_run > 1:
            with multiprocessing.Pool() as pool:
                items = [(sample_once_fn, fixed_kwargs)] * n_to_run
                for val in pool.imap_unordered(_invoke_sample_once, items, chunksize=1):
                    samples.append(val)
        else:
            for _ in range(n_to_run):
                samples.append(sample_once_fn(**fixed_kwargs))
        if _ci_halfwidth(samples) <= delta:
            break
    return float(np.mean(samples)), len(samples)

# Small helpers to read optional env overrides (safe defaults if unset)
def _env_steps(default_steps):
    v = os.environ.get("OEE_STEPS")
    try:
        return int(v) if v else default_steps
    except Exception:
        return default_steps

def _env_delta(default=0.02):
    v = os.environ.get("OEE_CI_DELTA", "")
    try:
        return float(v) if v else default
    except Exception:
        return default

def _env_batch(default=50):
    v = os.environ.get("OEE_CI_BATCH", "")
    try:
        return int(v) if v else default
    except Exception:
        return default

def _env_maxnets(default=1000):
    v = os.environ.get("OEE_MAX_NETS", "")
    try:
        return int(v) if v else default
    except Exception:
        return default
# ========================================================================

# ---- single-sample helpers (return one OEE sample) ----------------------
def dynamic_sample_once(K, num_nodes, topology, bias, steps, mutation_prob):
    nodes, functions = generate_boolean_network(num_nodes, K, topology, bias)
    nodes, functions, entangled = apply_dynamic_logic((nodes, functions), mutation_prob)
    states = simulate_pbn([(nodes, functions, entangled)], [1.0], steps, use_gpu=(USE_GPU and steps >= GPU_MIN_T), gpu_update=GPU_UPDATE)
    return compute_open_endedness(states, steps)

def paraconsistent_sample_once(K, num_nodes, topology, bias, steps, contradiction_prob):
    nodes, functions = generate_boolean_network(num_nodes, K, topology, bias)
    nodes, functions, entangled = apply_paraconsistent_logic((nodes, functions), contradiction_prob)
    states = simulate_pbn([(nodes, functions, entangled)], [1.0], steps, use_gpu=(USE_GPU and steps >= GPU_MIN_T), gpu_update=GPU_UPDATE)
    return compute_open_endedness(states, steps)

def modal_sample_once(K, num_nodes, topology, bias, steps, accessibility_degree, p_possible, p_necessary):
    nodes, functions = generate_boolean_network(num_nodes, K, topology, bias)
    nodes, functions, entangled, kripke = apply_modal_logic(
        (nodes, functions),
        accessibility_degree=accessibility_degree,
        p_possible=p_possible,
        p_necessary=p_necessary
    )
    states = simulate_pbn([(nodes, functions, entangled, kripke)], [1.0], steps, use_gpu=(USE_GPU and steps >= GPU_MIN_T), gpu_update=GPU_UPDATE)
    return compute_open_endedness(states, steps)

def quantum_sample_once(K, num_nodes, topology, bias, steps, superposition_prob, max_entangled):
    nodes, functions = generate_boolean_network(num_nodes, K, topology, bias)
    nodes, functions, entangled = apply_quantum_logic((nodes, functions), superposition_prob, max_entangled)
    states = simulate_pbn([(nodes, functions, entangled)], [1.0], steps, use_gpu=(USE_GPU and steps >= GPU_MIN_T), gpu_update=GPU_UPDATE)
    return compute_open_endedness(states, steps)
    
def pbn_sample_once(K, num_nodes, topology, bias, steps, switch_prob, num_contexts):
    """
    Build a PBN with `num_contexts` pure-Boolean contexts and compute one OEE sample.
    Uses measure_kd_piecewise (exact when all contexts are deterministic).
    """
    # Base RBN
    nodes, base_functions = generate_boolean_network(num_nodes, K, topology, bias)

    # Build additional contexts by resampling LUTs (pure Boolean, no entanglement)
    pbn_functions = [base_functions]
    for _ in range(num_contexts - 1):
        new_functions = {}
        for node, (regulators, lut) in base_functions.items():
            k_i = len(regulators)
            new_lut = {
                tuple(map(int, format(j, f'0{k_i}b'))): int(np.random.choice([0, 1]))
                for j in range(2**k_i)
            }
            new_functions[node] = (regulators, new_lut)
        pbn_functions.append(new_functions)

    # Uniform context probabilities
    probs = np.ones(num_contexts) / num_contexts

    # Exact piecewise if possible; otherwise falls back internally
    V, P, KD = measure_kd_piecewise(
        [(nodes, f, {}) for f in pbn_functions],
        probs,
        steps,
        switching=switch_prob,
        use_gpu=(USE_GPU and steps >= GPU_MIN_T),
        gpu_update=GPU_UPDATE)
    return compute_open_endedness((V, P, KD), steps)
# ------------------------------------------------------------------------

def compute_open_endedness(states_or_VPK, T):
    if isinstance(states_or_VPK, tuple) and len(states_or_VPK) == 3:
        V, P, KD = states_or_VPK
    else:
        V, P, KD = extract_attractor_metrics(states_or_VPK)
    return KD / T**2

def process_network(args):
    avg_connectivity, num_nodes, topology, bias, steps, num_contexts = args
    nodes, functions = generate_boolean_network(num_nodes, avg_connectivity, topology, bias)
    results = {model: [] for model in ["Original", "Paraconsistent", "Modal", "Quantum", "Dynamic", "PBN"]}

    # Original RBN (single deterministic context) — unchanged path
    states = simulate_pbn([(nodes, functions, {})], [1.0], steps,
                          use_gpu=(USE_GPU and steps >= GPU_MIN_T), gpu_update=GPU_UPDATE)
    results["Original"].append(compute_open_endedness(states, steps))

    transformations = {
        "Paraconsistent": apply_paraconsistent_logic,
        "Modal": apply_modal_logic,
        "Quantum": apply_quantum_logic,
        "Dynamic": apply_dynamic_logic
    }
    for name, transform in transformations.items():
        transformed_output = transform((nodes, functions, {}))
        if len(transformed_output) == 4:
            t_nodes, t_functions, t_entangled, t_kripke = transformed_output
            ctx = (t_nodes, t_functions, t_entangled, t_kripke)
        else:
            t_nodes, t_functions, t_entangled = transformed_output
            ctx = (t_nodes, t_functions, t_entangled)
        # Let measure_kd_piecewise decide (falls back if non-deterministic)
        VPK = measure_kd_piecewise([ctx], [1.0], steps, switching=0.0,
                                   use_gpu=(USE_GPU and steps >= GPU_MIN_T),
                                   gpu_update=GPU_UPDATE)
        results[name].append(compute_open_endedness(VPK, steps))

    # PBN (multi-context) — exact per-segment KD when all contexts are deterministic
    pbn_functions = [functions]
    for _ in range(num_contexts - 1):
        new_functions = {}
        for node, (regulators, lookup_table) in functions.items():
            k_i = len(regulators)
            new_lookup_table = {
                tuple(map(int, format(j, f'0{k_i}b'))): int(np.random.choice([0, 1]))
                for j in range(2**k_i)
            }
            new_functions[node] = (regulators, new_lookup_table)
        pbn_functions.append(new_functions)
    context_probabilities = np.ones(num_contexts) / num_contexts

    # Use the new exact piecewise method (falls back automatically if needed)
    VPK = measure_kd_piecewise([(nodes, f, {}) for f in pbn_functions],
                               context_probabilities, steps,
                               switching=0.5, use_gpu=(USE_GPU and steps >= GPU_MIN_T),
                               gpu_update=GPU_UPDATE)
    results["PBN"].append(compute_open_endedness(VPK, steps))

    return results

# Run Simulation for a Single Connectivity Value
def run_simulation(avg_connectivity, num_networks=1000, num_nodes=100, steps=10000):
    topology = "Exponential"
    bias = 0.5
    num_contexts = 5
    results = {model: [] for model in ["Original", "Paraconsistent", "Modal", "Quantum", "Dynamic", "PBN"]}

    # Track execution time
    start_time = time.time()

    # Sequential execution (no multiprocessing inside run_simulation)
    for _ in tqdm(range(num_networks), desc=f"Simulating Networks for K={avg_connectivity}", position=0, mininterval=0.5):
        network_results = process_network((avg_connectivity, num_nodes, topology, bias, steps, num_contexts))

        # Aggregate results
        for key, values in network_results.items():
            results[key].extend(values)

    execution_time = time.time() - start_time
    print(f"Execution time for K={avg_connectivity}: {execution_time:.2f} seconds")

    # Compute averages
    averaged_results = {key: np.mean(results[key]) for key in results}
    return avg_connectivity, averaged_results

# Helper function for parallelization
def parallel_run(args):
    K, switch_prob, num_contexts, num_networks, num_nodes, topology, bias, steps = args
    results = []
    for _ in range(num_networks):
        nodes, original_functions = generate_boolean_network(num_nodes, K, topology, bias)

        pbn_functions = [original_functions]
        for _ in range(num_contexts - 1):
            new_functions = {}
            for node, (regulators, lookup_table) in original_functions.items():
                k_i = len(regulators)
                new_lookup_table = {
                    tuple(map(int, format(j, f'0{k_i}b'))): np.random.choice([0, 1])
                    for j in range(2**k_i)
                }
                new_functions[node] = (regulators, new_lookup_table)
            pbn_functions.append(new_functions)

        context_probabilities = np.ones(num_contexts) / num_contexts
        states = simulate_pbn(
            [(nodes, f, {}) for f in pbn_functions],
            context_probabilities, steps, switching=switch_prob,
            use_gpu=(USE_GPU and steps >= GPU_MIN_T),
            gpu_update=GPU_UPDATE, pbn_dependent=PBN_DEP
        )

        OEE = compute_open_endedness(states, steps)
        results.append(OEE)

    return np.mean(results)

#Quantum logic runner
def quantum_run(args):
    K, num_networks, num_nodes, topology, bias, steps, superposition_prob, max_entangled = args
    results = []
    for _ in range(num_networks):
        nodes, functions = generate_boolean_network(num_nodes, K, topology, bias)
        nodes, functions, entangled_pairs = apply_quantum_logic((nodes, functions), superposition_prob, max_entangled)
        states = simulate_pbn([(nodes, functions, entangled_pairs)], [1.0], steps,
                      use_gpu=(USE_GPU and steps >= GPU_MIN_T), gpu_update=GPU_UPDATE)
        results.append(compute_open_endedness(states, steps))
    return np.mean(results)

# Dynamic logic runner
def dynamic_run(args):
    K, num_networks, num_nodes, topology, bias, steps, mutation_prob = args
    results = []
    for _ in range(num_networks):
        nodes, functions = generate_boolean_network(num_nodes, K, topology, bias)
        nodes, functions, entangled_pairs = apply_dynamic_logic((nodes, functions), mutation_prob)
        states = simulate_pbn([(nodes, functions, entangled_pairs)], [1.0], steps,
                      use_gpu=(USE_GPU and steps >= GPU_MIN_T), gpu_update=GPU_UPDATE)
        results.append(compute_open_endedness(states, steps))
    return np.mean(results)

# ─── Paraconsistent logic runner ─────────────────────────────────────────
def paraconsistent_run(args):
    """
    args = (K, num_networks, num_nodes, topology, bias, steps, contradiction_prob)
    """
    K, num_networks, num_nodes, topology, bias, steps, contradiction_prob = args
    results = []
    for _ in range(num_networks):
        # 1) generate base RBN
        nodes, functions = generate_boolean_network(num_nodes, K, topology, bias)
        # 2) inject paraconsistent logic
        nodes, functions, entangled_pairs = apply_paraconsistent_logic(
            (nodes, functions), contradiction_prob
        )
        # 3) simulate
        states = simulate_pbn([(nodes, functions, entangled_pairs)], [1.0], steps,
                      use_gpu=(USE_GPU and steps >= GPU_MIN_T), gpu_update=GPU_UPDATE)
        results.append(compute_open_endedness(states, steps))
    return np.mean(results)


# ─── Modal logic runner ───────────────────────────────────────────────────
def modal_run(args):
    """
    args = (K, num_networks, num_nodes, topology, bias, steps,
            accessibility_degree, p_possible, p_necessary)
    """
    (K, num_networks, num_nodes, topology, bias, steps,
     accessibility_degree, p_possible, p_necessary) = args

    results = []
    for _ in range(num_networks):
        # 1) generate base RBN
        nodes, functions = generate_boolean_network(num_nodes, K, topology, bias)
        # 2) inject modal logic
        nodes, functions, entangled_pairs, kripke_frame = apply_modal_logic(
            (nodes, functions),
            accessibility_degree=accessibility_degree,
            p_possible=p_possible,
            p_necessary=p_necessary
        )
        # 3) simulate (include kripke_frame so simulate_pbn can use it)
        states = simulate_pbn(
	    [(nodes, functions, entangled_pairs, kripke_frame)],
	    [1.0],
	    steps,
	    use_gpu=(USE_GPU and steps >= GPU_MIN_T), gpu_update=GPU_UPDATE
	)

        results.append(compute_open_endedness(states, steps))
    return np.mean(results)

# Parallelized main function

def main():
    avg_connectivities = np.arange(1.1, 4.7, 0.2)
    num_nodes = 100
    topology  = "Exponential"
    bias      = 0.5

    steps = _env_steps(100000)     # optional: export OEE_STEPS=50000 to halve T
    delta = _env_delta(0.02)       # OEE_CI_DELTA (95% CI half-width)
    batch = _env_batch(50)         # OEE_CI_BATCH
    max_n = _env_maxnets(1000)     # cap matches your original 1000 nets

    switching_probabilities = [0.1, 0.3, 0.5, 0.7, 0.9]
    num_contexts_list       = [2, 4, 6, 8, 10]

    final_results = {}

    for switch_prob in switching_probabilities:
        for num_contexts in num_contexts_list:
            label = f"SwitchProb={switch_prob}_Contexts={num_contexts}"
            vals = []
            for K in tqdm(avg_connectivities, desc=f"PBN {label}"):
                mean_oee, n_used = _run_until_ci(
                    pbn_sample_once,
                    dict(K=K, num_nodes=num_nodes, topology=topology, bias=bias, steps=steps,
                         switch_prob=switch_prob, num_contexts=num_contexts),
                    batch=batch, max_n=max_n, delta=delta, use_pool=True
                )
                vals.append(mean_oee)
            final_results[label] = vals

    np.save("open_endedness_pbn.npy", final_results)

    # Plot results (unchanged style/filenames)
    plt.figure(figsize=(10, 6))
    for label, values in final_results.items():
        plt.plot(avg_connectivities, values, label=label)
    plt.xlabel("Avg Connectivity (K)")
    plt.ylabel("Open-endedness")
    plt.legend()
    plt.title("Parallelized Open-endedness Analysis")
    plt.savefig("open_endedness_pbn.png")
    if os.environ.get("OEE_SHOW_PLOTS", "0") == "1":
        plt.show()
    else:
        plt.close()


#################################################

def main_quantum_logic():
    avg_connectivities = np.arange(1.1, 4.7, 0.2)
    num_nodes = 100
    topology  = "Exponential"
    bias      = 0.5
    steps     = _env_steps(100000)
    delta     = _env_delta(0.02)
    batch     = _env_batch(50)
    max_n     = _env_maxnets(1000)

    superposition_probs = [0.1, 0.3, 0.5, 0.7, 0.9]
    entangled_sizes     = [2, 4, 6, 8, 16, 32]

    final_results = {}
    for p in superposition_probs:
        for ent in entangled_sizes:
            label = f"Superposition={p}_Entangled={ent}"
            vals  = []
            for K in tqdm(avg_connectivities, desc=f"Quantum {label}"):
                mean_oee, n_used = _run_until_ci(
                    quantum_sample_once,
                    dict(K=K, num_nodes=num_nodes, topology=topology, bias=bias, steps=steps,
                         superposition_prob=p, max_entangled=ent),
                    batch=batch, max_n=max_n, delta=delta, use_pool=True
                )
                vals.append(mean_oee)
            final_results[label] = vals

    np.save("open_endedness_quantum.npy", final_results)

    plt.figure(figsize=(10, 6))
    for label, values in final_results.items():
        plt.plot(avg_connectivities, values, label=label)
    plt.xlabel("Avg Connectivity (K)")
    plt.ylabel("Open-endedness")
    plt.title("Quantum Logic Parameter Sweep")
    plt.legend()
    plt.savefig("open_endedness_quantum.png")
    if os.environ.get("OEE_SHOW_PLOTS", "0") == "1":
        plt.show()
    else:
        plt.close()

#######################################################

def main_dynamic_logic():
    avg_connectivities = np.arange(1.1, 4.7, 0.2)
    num_nodes = 100
    topology  = "Exponential"
    bias      = 0.5
    steps     = _env_steps(100000)     # optional override via OEE_STEPS
    delta     = _env_delta(0.02)       # OEE_CI_DELTA
    batch     = _env_batch(50)         # OEE_CI_BATCH
    max_n     = _env_maxnets(1000)     # OEE_MAX_NETS

    mutation_probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    final_results = {}

    for mu in mutation_probs:
        label = f"MutationProb={mu}"
        vals  = []
        for K in tqdm(avg_connectivities, desc=f"Dynamic {label}"):
            mean_oee, n_used = _run_until_ci(
                dynamic_sample_once,
                dict(K=K, num_nodes=num_nodes, topology=topology, bias=bias, steps=steps, mutation_prob=mu),
                batch=batch, max_n=max_n, delta=delta, use_pool=True
            )
            vals.append(mean_oee)
        final_results[label] = vals

    np.save("open_endedness_dynamic.npy", final_results)

    plt.figure(figsize=(10, 6))
    for label, values in final_results.items():
        plt.plot(avg_connectivities, values, label=label)
    plt.xlabel("Avg Connectivity (K)")
    plt.ylabel("Open-endedness")
    plt.title("Dynamic Logic Parameter Sweep")
    plt.legend()
    plt.savefig("open_endedness_dynamic.png")
    if os.environ.get("OEE_SHOW_PLOTS", "0") == "1":
        plt.show()
    else:
        plt.close()

########################################################

def main_paraconsistent_logic():
    avg_connectivities = np.arange(1.1, 4.7, 0.2)
    num_nodes = 100
    topology  = "Exponential"
    bias      = 0.5
    steps     = _env_steps(100000)
    delta     = _env_delta(0.02)
    batch     = _env_batch(50)
    max_n     = _env_maxnets(1000)

    contradiction_probs = [0.1, 0.3, 0.5, 0.7, 0.9]
    final_results = {}

    for p in contradiction_probs:
        label = f"ContrProb={p}"
        vals  = []
        for K in tqdm(avg_connectivities, desc=f"Paraconsistent {label}"):
            mean_oee, n_used = _run_until_ci(
                paraconsistent_sample_once,
                dict(K=K, num_nodes=num_nodes, topology=topology, bias=bias, steps=steps, contradiction_prob=p),
                batch=batch, max_n=max_n, delta=delta, use_pool=True
            )
            vals.append(mean_oee)
        final_results[label] = vals

    np.save("open_endedness_paraconsistent.npy", final_results)

    plt.figure(figsize=(10, 6))
    for label, values in final_results.items():
        plt.plot(avg_connectivities, values, label=label)
    plt.xlabel("Avg Connectivity (K)")
    plt.ylabel("Open-endedness")
    plt.title("Paraconsistent Logic Parameter Sweep")
    plt.legend()
    plt.savefig("open_endedness_paraconsistent.png")
    if os.environ.get("OEE_SHOW_PLOTS", "0") == "1":
        plt.show()
    else:
        plt.close()

#########################################################
    
def main_modal_logic():
    avg_connectivities = np.arange(1.1, 4.7, 0.2)
    num_nodes = 100
    topology  = "Exponential"
    bias      = 0.5
    steps     = _env_steps(100000)
    delta     = _env_delta(0.02)
    batch     = _env_batch(50)
    max_n     = _env_maxnets(1000)

    accessibility_degrees = [1, 5, 10]
    p_possibilities       = [0.1, 0.5, 0.9]
    p_necessities         = [0.1, 0.5, 0.9]

    final_results = {}
    for d in accessibility_degrees:
        for p_poss in p_possibilities:
            for p_necc in p_necessities:
                label = f"Access={d}_Pposs={p_poss}_Pnecc={p_necc}"
                vals  = []
                for K in tqdm(avg_connectivities, desc=f"Modal {label}"):
                    mean_oee, n_used = _run_until_ci(
                        modal_sample_once,
                        dict(K=K, num_nodes=num_nodes, topology=topology, bias=bias, steps=steps,
                             accessibility_degree=d, p_possible=p_poss, p_necessary=p_necc),
                        batch=batch, max_n=max_n, delta=delta, use_pool=True
                    )
                    vals.append(mean_oee)
                final_results[label] = vals

    np.save("open_endedness_modal.npy", final_results)

    plt.figure(figsize=(10, 6))
    for label, values in final_results.items():
        plt.plot(avg_connectivities, values, label=label)
    plt.xlabel("Avg Connectivity (K)")
    plt.ylabel("Open-endedness")
    plt.title("Modal Logic Parameter Sweep")
    plt.legend()
    plt.savefig("open_endedness_modal.png")
    if os.environ.get("OEE_SHOW_PLOTS", "0") == "1":
        plt.show()
    else:
        plt.close()

'''
# ---------- QUICK SMOKE TEST --------------------------------------------
def quick_sanity_check():
    """
    Fast surrogate for the 3.5-hour run.
    ─────────────────────────────────────────
    •  K grid       : 3 values that hit ordered / critical / chaotic
    •  networks     : 40 (instead of 1 000)
    •  contexts     : 5  (same as main experiment)
    •  run length   : 5 000 steps
    •  aggregation  : median  (more robust for tiny sample)
    Produces a figure and prints medians so you can see immediately
    whether the PBN belly is present before committing to the full job.
    """
    K_grid        = [1.4, 2.0, 3.0]          # ≈ ordered, critical, chaotic
    num_networks  = 40
    steps         = 5000                    # 5k steps = < 1 s per net
    num_contexts  = 5
    topology      = "Poisson"
    bias          = 0.5

    models  = ["Original", "Paraconsistent", "Modal",
               "Quantum", "Dynamic", "PBN"]
    results = {m: [] for m in models}

    for K in K_grid:
        # reuse existing machinery
        _, avg = run_simulation(K,
                                num_networks=num_networks,
                                num_nodes   =100,
                                steps       =steps)
        for m in models:
            results[m].append(avg[m])

    # ---------------- plot -----------------
    colours = dict(zip(models, ["b", "r", "g", "m", "c", "y"]))  # simple colour map

    plt.figure(figsize=(6,4))
    for m in models:
        plt.plot(K_grid, results[m], label=m, color=colours[m])
    plt.xlabel("Avg connectivity  K")
    plt.ylabel("Open-endedness  Ω")
    plt.legend(); plt.tight_layout()
    plt.title("SMOKE TEST  •  {} nets  •  {} steps".format(num_networks, steps))
    plt.show()

    # print raw medians for eyeballing
    for m in models:
        print(f"{m:13s}", results[m])
# ------------------------------------------------------------------------
'''
################################################################################

# Run the full pipeline
if __name__ == "__main__":
    #np.random.seed(0)
    #quick_sanity_check()
    #main_dynamic_logic()
    main()
    main_paraconsistent_logic()
    main_quantum_logic()
    main_modal_logic()
'''  
    avg_connectivities = np.arange(1.1, 4.7, 0.2)
    final_results = {}

    # Run `run_simulation` in parallel across different `avg_connectivity` values
    with multiprocessing.Pool() as pool:
        results = list(tqdm(pool.starmap(run_simulation, [(k,) for k in avg_connectivities], chunksize=2), 
                            total=len(avg_connectivities), desc="Processing Avg Connectivity"))

    for k, avg_values in results:
        final_results[k] = avg_values

    np.save("open_endedness_results.npy", final_results)

    # Plot Results
    models = ["Original", "Paraconsistent", "Modal", "Quantum", "Dynamic", "PBN"]
    colors = ["b", "r", "g", "m", "c", "y"]

    plt.figure(figsize=(8, 5))
    
    for model, color in zip(models, colors):
        x = list(final_results.keys())
        y = [final_results[k][model] for k in x]
        plt.plot(x, y, label=model, color=color)

    plt.xlabel("Avg Connectivity (K)")
    plt.ylabel("Open-endedness")
    #plt.yscale('log')
    plt.legend()
    plt.savefig("open_endedness_plot.png")
    plt.show()
'''
