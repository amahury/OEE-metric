import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 1200
import multiprocessing
from tqdm import tqdm
from axiomatic import generate_boolean_network, simulate_pbn
import os

# Toggle + thresholds (env-overridable)
USE_GPU = os.environ.get("OEE_USE_GPU", "1") != "0"
GPU_MIN_T = int(os.environ.get("OEE_GPU_MIN_T", "200000"))   # when T >= this, prefer GPU streaming
GPU_UPDATE = os.environ.get("OEE_GPU_UPDATE", "synchronous")
GPU_CHUNK = os.environ.get("OEE_GPU_CHUNK_T")
GPU_CHUNK = int(GPU_CHUNK) if GPU_CHUNK is not None else None

def compute_total_variety(attractors):
    return sum(len(attractor) for attractor in attractors)

def compute_total_persistence(attractors):
    return sum(t2 - t1 for t1, t2 in attractors)

def compute_total_basin_length(attractors):
    basin_lengths = []
    prev_exit_time = 0
    for entry_time, exit_time in attractors:
        basin_lengths.append(entry_time - prev_exit_time)
        prev_exit_time = exit_time
    return sum(basin_lengths)

def extract_attractor_metrics(network_states):
    """
    Returns (V, P, KD). Works with a NumPy array (T,N) *or* any iterable of states.
    """
    if isinstance(network_states, np.ndarray):
        packed = np.packbits(network_states.astype(np.uint8), axis=1)
        seen, cycle_len, dwell = {}, {}, {}
        current = None
        for t in range(packed.shape[0]):
            s_bytes = packed[t].tobytes()
            if current is None:
                if s_bytes in seen:
                    entry = seen[s_bytes]
                    k = t - entry
                    current = entry
                    cycle_len[current] = k
                    dwell[current] = 1
                else:
                    seen[s_bytes] = t
            else:
                dwell[current] += 1
                if s_bytes not in seen:
                    current = None
                seen[s_bytes] = t
        V = sum(cycle_len.values())
        P = sum(dwell.values())
        KD = sum(cycle_len[a] * dwell[a] for a in cycle_len)
        return V, P, KD

    # generic streaming path over an iterable
    seen, cycle_len, dwell = {}, {}, {}
    current = None
    for t, s in enumerate(map(tuple, network_states)):
        if current is None:
            if s in seen:
                entry = seen[s]
                k = t - entry
                current = entry
                cycle_len[current] = k
                dwell[current] = 1
            else:
                seen[s] = t
        else:
            dwell[current] += 1
            if s not in seen:
                current = None
            seen[s] = t
    V = sum(cycle_len.values())
    P = sum(dwell.values())
    KD = sum(cycle_len[a] * dwell[a] for a in cycle_len)
    return V, P, KD

def simulate_rbn_and_measure_metrics(num_nodes, avg_connectivity, topology, bias, steps):
    nodes, functions = generate_boolean_network(num_nodes, avg_connectivity, topology, bias)
    use_gpu_now = (USE_GPU and steps >= GPU_MIN_T)
    network_states = simulate_pbn(
        [(nodes, functions, {})],
        [1.0],
        steps,
        use_gpu=use_gpu_now,
        gpu_update=GPU_UPDATE,
    )
    V, P, KD = extract_attractor_metrics(network_states)
    return KD / steps**2

def simulate_rbn_and_measure_metrics_cycle_tail(num_nodes, avg_connectivity, topology, bias, steps, use_gpu=False):
    """
    Exact long-T OEE for a single deterministic BN by stopping once the first cycle is found,
    then filling the tail analytically: V=lambda, P=T-mu, KD=lambda*(T-mu).
    """
    from axiomatic import generate_boolean_network, simulate_pbn
    nodes, functions = generate_boolean_network(num_nodes, avg_connectivity, topology, bias)

    # Try GPU if available; fall back gracefully if simulate_pbn has no GPU args
    try:
        gen = simulate_pbn([(nodes, functions, {})], [1.0], steps,
                           use_gpu=use_gpu, gpu_update='synchronous')
    except TypeError:
        gen = simulate_pbn([(nodes, functions, {})], [1.0], steps)

    seen = {}
    mu = lam = None
    for t, s in enumerate(gen):
        st = tuple(s)
        if st in seen:
            mu = seen[st]
            lam = t - mu
            break
        seen[st] = t

    if mu is None or lam is None:
        # No repeat detected within the window (rare on smaller N; can happen).
        # Fall back to your existing exact finite-window extractor.
        try:
            states = simulate_pbn([(nodes, functions, {})], [1.0], steps)
        except TypeError:
            states = simulate_pbn([(nodes, functions, {})], [1.0], steps)
        V, P, KD = extract_attractor_metrics(states)
        return KD / steps**2

    V = lam
    P = max(0, steps - mu)
    KD = V * P
    return KD / (steps**2)
    
def process_networks(avg_connectivity, num_networks, num_nodes, topology, bias, steps):
    LONG_T = int(os.environ.get("OEE_LONG_T_THRESHOLD", "1000000"))
    use_gpu = os.environ.get("OEE_USE_GPU", "1") != "0"

    if steps >= LONG_T:
        vals = []
        for _ in range(num_networks):
            vals.append(simulate_rbn_and_measure_metrics_cycle_tail(
                num_nodes, avg_connectivity, topology, bias, steps, use_gpu=use_gpu))
        return np.mean(vals)

    # original fast CPU parallel path for modest T
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.starmap(
            simulate_rbn_and_measure_metrics,
            [(num_nodes, avg_connectivity, topology, bias, steps) for _ in range(num_networks)]
        )
    return np.mean(results)

'''
def process_networks(avg_connectivity, num_networks, num_nodes, topology, bias, steps):
    """
    For large T (GPU mode), avoid spawning many processes (single GPU contention).
    For small T, keep your original CPU parallel loop.
    """
    use_gpu_now = (USE_GPU and steps >= GPU_MIN_T)
    if use_gpu_now:
        vals = []
        for _ in range(num_networks):
            vals.append(simulate_rbn_and_measure_metrics(num_nodes, avg_connectivity, topology, bias, steps))
        return np.mean(vals)

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.starmap(
            simulate_rbn_and_measure_metrics,
            [(num_nodes, avg_connectivity, topology, bias, steps) for _ in range(num_networks)]
        )
    return np.mean(results)
 '''

def main():
    num_networks = 1000
    num_nodes = 100
    topology = "Poisson"
    bias = 0.5

    # Use your preferred T list. (Large Ts will stream on GPU.)
    ##timesteps_list = [10, 100, 1000, 10000, 100000]  # You can re-enable your huge T once this works
    timesteps_list = [1000000]  # example large run
    k_values = np.arange(1.1, 4.7, 0.2)

    results = {T: [] for T in timesteps_list}
    for avg_connectivity in tqdm(k_values, desc="Processing Connectivity Values"):
        for T in tqdm(timesteps_list, desc=f"K={avg_connectivity:.1f}", leave=False):
            avg_open_endedness = process_networks(avg_connectivity, num_networks, num_nodes, topology, bias, T)
            results[T].append(avg_open_endedness)

    for T in timesteps_list:
        np.save(f"open_endedness_T{T}.npy", results[T])

    plt.figure(figsize=(8, 6))
    for T in timesteps_list:
        plt.plot(k_values, results[T], label=f"T={T}")
    plt.xlabel("Average Connectivity (K)")
    plt.ylabel("Open-endedness")
    plt.legend()
    plt.title("Nets=1000; Nodes=100")
    plt.show()

if __name__ == "__main__":
    main()
