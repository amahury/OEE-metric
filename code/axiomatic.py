import numpy as np
import networkx as nx
from itertools import combinations, product
import random
import zlib
from scipy.stats import zipf, expon, poisson
from itertools import zip_longest
import multiprocessing
from functools import partial
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 1200
from functools import partial
import os
from cubewalkers import update_schemes as cw_update

# --- Optional GPU backends (cubewalkers) ---
try:
    import cupy as cp
    from cubewalkers import update_schemes as cw_update
    from cubewalkers import parser as cw_parser
    from cubewalkers import simulation as cw_sim
    _CUBEWALKERS_AVAILABLE = True
except Exception:
    cp = None
    cw_update = None
    cw_parser = None
    cw_sim = None
    _CUBEWALKERS_AVAILABLE = False

# ---- Optional CUDA block size overrides via env (safe defaults) ----
_OEE_TPB_X = int(os.environ.get("OEE_GPU_TPB_X", "64"))
_OEE_TPB_Y = int(os.environ.get("OEE_GPU_TPB_Y", "1"))

class KripkeFrame:
    def __init__(self, states):
        self.states = states
        self.accessibility = {state: set() for state in states}
    def add_accessibility(self, from_state, to_state):
        self.accessibility[from_state].add(to_state)
    def get_accessible_states(self, state):
        return self.accessibility.get(state, set())

# ===== your metrics/helpers (unchanged) =====
def kolmogorov_complexity(state):
    binary_string = ''.join(map(str, state))
    compressed = zlib.compress(binary_string.encode('utf-8'))
    return len(compressed)

def sophistication(network_states):
    complexities = np.array([kolmogorov_complexity(state) for state in network_states])
    mean_complexity = np.mean(complexities)
    random_baseline = np.min(complexities)
    return mean_complexity - random_baseline

def coarse_sophistication(network_states):
    complexities = np.array([kolmogorov_complexity(state) for state in network_states])
    complexity_differences = np.diff(complexities)
    penalty = np.sum(complexity_differences == 0)
    return max(0, np.mean(complexities) - np.min(complexities) - penalty)

def busy_beaver_logical_depth(network_states):
    depth_scores = []
    for prev_state, state in zip_longest(network_states[:-1], network_states[1:], fillvalue=[]):
        if not prev_state:
            continue
        changes = sum(p != s for p, s in zip(prev_state, state))
        depth_scores.append(changes)
    return np.sum(depth_scores)

def generate_boolean_network(num_nodes, avg_connectivity, topology, bias):
    nodes = [f"x{i}" for i in range(num_nodes)]
    functions = {}

    if topology == "Zipf":
        degree_distribution = zipf.rvs(2, size=num_nodes)
    elif topology == "Exponential":
        degree_distribution = expon.rvs(scale=avg_connectivity, size=num_nodes).astype(int)
    elif topology == "Poisson":
        degree_distribution = poisson.rvs(mu=avg_connectivity, size=num_nodes)
    else:
        raise ValueError("Invalid topology choice")

    degree_distribution = np.clip(degree_distribution, 1, num_nodes - 1)

    # >>> NEW: cap indegree to keep LUT sizes sane (default 8; override with OEE_MAX_K)
    K_CAP = int(os.environ.get("OEE_MAX_K", "8"))

    for i, node in enumerate(nodes):
        # cap indegree by both N-1 and K_CAP
        k_i = min(int(degree_distribution[i]), num_nodes - 1, K_CAP)

        regulators = random.sample(nodes, k_i)
        # truth table has size 2^k_i; with k_i<=8 this is at most 256 entries
        lookup_table = {
            tuple(map(int, format(j, f'0{k_i}b'))): (1 if random.random() < bias else 0)
            for j in range(2 ** k_i)
        }
        functions[node] = (regulators, lookup_table)

    return nodes, functions

# ===== your alternative logics (kept) =====
def apply_paraconsistent_logic(network, contradiction_prob=0.3):
    if len(network) == 3:
        nodes, functions, entangled_pairs = network
    else:
        nodes, functions = network
        entangled_pairs = {}
    updated_functions = {}
    for node, (regs, lut) in functions.items():
        newlut = {}
        for key, value in lut.items():
            if random.random() < contradiction_prob:
                newlut[key] = (2, value)
            else:
                newlut[key] = value
        updated_functions[node] = (regs, newlut)
    return nodes, updated_functions, entangled_pairs

def propagate_paraconsistent_state(value):
    if isinstance(value, tuple) and value[0] == 2:
        return value
    return value

def resolve_contradiction(inputs):
    values = [v for v in inputs if v != 2]
    if not values:
        return (2, 0)
    unique_values = set(values)
    if len(unique_values) == 1:
        return unique_values.pop()
    else:
        return (2, 1 if values.count(1) >= values.count(0) else 0)

def apply_modal_logic(network, accessibility_degree=None, p_possible=0.3, p_necessary=0.3,
                      nec_strength_range=(0.7, 1.0)):
    if len(network) == 2:
        nodes, functions = network
        entangled_pairs = {}
        kripke_frame = KripkeFrame(states=nodes)
    elif len(network) == 3:
        nodes, functions, entangled_pairs = network
        kripke_frame = KripkeFrame(states=nodes)
    else:
        nodes, functions, entangled_pairs, kripke_frame = network
    d = accessibility_degree or min(len(nodes)//3, 3)
    for node in nodes:
        accessible_nodes = random.sample(nodes, min(d, len(nodes)))
        for acc_node in accessible_nodes:
            kripke_frame.add_accessibility(node, acc_node)
    updated_functions = {}
    for node, (regs, lut) in functions.items():
        newlut = {}
        for key, value in lut.items():
            accessible_states = [lut.get(tuple(random.choice(list(lut.keys()))), 0)
                                 for _ in kripke_frame.get_accessible_states(node)]
            accessible_int_values = [v[1] if isinstance(v, tuple) and v[0] == 2 else v
                                     for v in accessible_states]
            if isinstance(value, tuple) and value[0] == 2:
                value = value[1]
            if random.random() < p_possible and accessible_int_values:
                newlut[key] = "possible"
            elif random.random() < p_necessary and accessible_int_values:
                necessity_value = max(accessible_int_values + [value])
                newlut[key] = ("necessary", necessity_value,
                               random.uniform(nec_strength_range[0], nec_strength_range[1]))
            else:
                newlut[key] = value
        updated_functions[node] = (regs, newlut)
    return nodes, updated_functions, entangled_pairs, kripke_frame

def resolve_modal_state(value, context_factor, accessible_values):
    if value == "superposed":
        return collapse_superposition(value)
    elif value == "possible":
        return 1 if any(accessible_values) else 0
    elif isinstance(value, tuple) and value[0] == "necessary":
        necessary_value = value[1]
        if isinstance(necessary_value, tuple) and necessary_value[0] == 2:
            necessary_value = necessary_value[1]
        return necessary_value if all(accessible_values) else 0
    return value

def apply_quantum_logic(network, superposition_prob=0.3, max_entangled_pairs=5):
    if len(network) == 4:
        nodes, functions, entangled_pairs, kripke_frame = network
    elif len(network) == 3:
        nodes, functions, entangled_pairs = network
        kripke_frame = None
    else:
        nodes, functions = network
        entangled_pairs = {}
        kripke_frame = None
    updated_functions = {}
    for node, (regs, lut) in functions.items():
        newlut = {}
        for key, value in lut.items():
            if random.random() < superposition_prob:
                newlut[key] = "superposed"
            else:
                newlut[key] = value
        updated_functions[node] = (regs, newlut)
    if not entangled_pairs:
        entanglement_nodes = random.sample(nodes, min(len(nodes), max_entangled_pairs * 2))
        for i in range(0, len(entanglement_nodes) - 1, 2):
            entangled_pairs[entanglement_nodes[i]] = entanglement_nodes[i + 1]
            entangled_pairs[entanglement_nodes[i + 1]] = entanglement_nodes[i]
    if kripke_frame:
        return nodes, updated_functions, entangled_pairs, kripke_frame
    return nodes, updated_functions, entangled_pairs

def collapse_superposition(value):
    if value == "superposed":
        return random.choice([0, 1])
    return value

def mutate_rule(value):
    if isinstance(value, dict) and "mutation_probability" in value:
        if isinstance(value["initial_value"], int):
            if random.random() < value["mutation_probability"]:
                return 1 - value["initial_value"]
            return value["initial_value"]
        else:
            return value["initial_value"]
    return value

def apply_dynamic_logic(network, mutation_prob=0.5):
    if len(network) == 4:
        nodes, functions, entangled_pairs, kripke_frame = network
    elif len(network) == 3:
        nodes, functions, entangled_pairs = network
        kripke_frame = None
    else:
        nodes, functions = network
        entangled_pairs = {}
        kripke_frame = None
    updated_functions = {}
    for node, (regs, lut) in functions.items():
        newlut = {}
        for key, value in lut.items():
            if isinstance(value, int):
                newlut[key] = {"initial_value": value, "mutation_probability": mutation_prob}
            else:
                newlut[key] = value
        updated_functions[node] = (regs, newlut)
    if kripke_frame:
        return nodes, updated_functions, entangled_pairs, kripke_frame
    return nodes, updated_functions, entangled_pairs

# ============= GPU helpers (unchanged) =============
def _is_pure_boolean_functions(functions):
    for _, (_, lut) in functions.items():
        for v in lut.values():
            if not isinstance(v, (int, np.integer, bool, np.bool_)):
                return False
    return True

def _all_contexts_pure_boolean(networks):
    for ctx in networks:
        if len(ctx) == 4:
            _, functions, entangled_pairs, _ = ctx
        else:
            _, functions, entangled_pairs = ctx
        if entangled_pairs:
            return False
        if not _is_pure_boolean_functions(functions):
            return False
    return True

def _build_regulators_and_lut(nodes, functions):
    name2idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)
    k_list = [len(functions[n][0]) for n in nodes]
    kmax = max(k_list) if k_list else 0
    max_rows = 1 << kmax
    merged = np.zeros((max_rows, N), dtype=np.bool_)
    node_regs = []
    for j, node in enumerate(nodes):
        regs, lut = functions[node]
        idxs = [name2idx[r] for r in regs]
        node_regs.append(idxs)
        k = len(regs)
        if k == 0:
            merged[0, j] = bool(lut.get((), 0))
            continue
        for irow in range(1 << k):
            key = tuple(map(int, format(irow, f'0{k}b')))
            merged[irow, j] = bool(lut.get(key, 0))
    return node_regs, merged

def _estimate_gpu_chunk_T(N, bytes_per_val=1, max_frac=0.20, hard_cap=1_000_000):
    if not (_CUBEWALKERS_AVAILABLE and cp is not None):
        return min(hard_cap, 100_000)
    try:
        free_bytes, _total_bytes = cp.cuda.runtime.memGetInfo()
        target = int(free_bytes * max_frac)
        T = max(1, target // max(N, 1) // bytes_per_val)
        return int(min(T, hard_cap))
    except Exception:
        return min(hard_cap, 100_000)
        
def _choose_cw_mask(update: str, is_pbn: bool, pbn_dependent: bool) :
    """
    update ∈ {"synchronous","asynchronous","asynchronous_set"}
    is_pbn: use PBN-aware mask variants
    pbn_dependent: use *dependent* variants that share a single PBN value across nodes (per walker/step)
    """
    if not is_pbn:
        if update == "synchronous":
            return cw_update.synchronous
        elif update == "asynchronous":
            return cw_update.asynchronous
        elif update == "asynchronous_set":
            return cw_update.asynchronous_set
        else:
            raise ValueError(f"Unknown update '{update}' for non-PBN.")
    else:
        if update == "synchronous":
            return cw_update.synchronous_PBN_dependent if pbn_dependent else cw_update.synchronous_PBN
        elif update == "asynchronous":
            # single-node update with PBN support (independent per node)
            return cw_update.asynchronous_PBN
        elif update == "asynchronous_set":
            # subset update with a *shared* PBN value across nodes if dependent=True
            return cw_update.asynchronous_set_PBN_dependent if pbn_dependent else cw_update.asynchronous_set_PBN
        else:
            raise ValueError(f"Unknown update '{update}' for PBN.")

def _gpu_state_stream_for_context(nodes, functions, initial_state_np, steps,
                                  gpu_update='synchronous', is_pbn=False, pbn_dependent=True,
                                  chunk_T=None, threads_per_block=None):
    assert _CUBEWALKERS_AVAILABLE, "cubewalkers not available"
    if chunk_T is None:
        chunk_T = _estimate_gpu_chunk_T(len(nodes))
    if threads_per_block is None:
        tpb_x = min(_OEE_TPB_X, 128)
        tpb_y = max(1, _OEE_TPB_Y)
        threads_per_block = (tpb_x, tpb_y)

    node_regs, merged_lut_np = _build_regulators_and_lut(nodes, functions)
    kernel, _ = cw_parser.regulators2lutkernel(node_regs, kernel_name="lut_kernel")
    merged_lut = cp.asarray(merged_lut_np, dtype=cp.bool_)
    maskfunction = _choose_cw_mask(gpu_update, is_pbn, pbn_dependent)
    curr = cp.asarray(np.asarray(initial_state_np, dtype=np.bool_), dtype=cp.bool_).reshape(len(nodes), 1)
    steps_left = int(steps)
   
    while steps_left > 0:
        Tseg = int(min(steps_left, chunk_T))
        traj = cw_sim.simulate_ensemble(
            kernel=kernel, N=len(nodes), T=Tseg, W=1, T_window=None,
            lookup_tables=merged_lut, averages_only=False,
            initial_states=curr, maskfunction=maskfunction,
            threads_per_block=threads_per_block
        )
        arr = cp.asnumpy(traj)      # Tseg x N x 1
        del traj
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass
        for row in arr:
            yield row[:, 0].astype(np.uint8).tolist()
        curr = cp.asarray(arr[-1, :, 0], dtype=cp.bool_).reshape(len(nodes), 1)
        steps_left -= Tseg

def _gpu_state_stream_PBN(networks, probabilities, steps, switching,
                          gpu_update='synchronous', pbn_dependent=True, chunk_T=None):
    num_contexts = len(networks)
    current_context = np.random.choice(num_contexts, p=probabilities)
    net = networks[current_context]
    if len(net) == 4:
        nodes, functions, entangled_pairs, _kr = net
    else:
        nodes, functions, entangled_pairs = net
        _kr = KripkeFrame(states=nodes)
    init = [random.choice([0, 1]) for _ in nodes]
    yield init
    steps_left = int(steps)
    while steps_left > 0:
        seg = steps_left if switching <= 0 else min(int(np.random.geometric(p=switching)), steps_left)
        for s in _gpu_state_stream_for_context(nodes, functions, init, seg,
                                               gpu_update=gpu_update, is_pbn=True,
                                               pbn_dependent=pbn_dependent,
                                               chunk_T=chunk_T,
                                               threads_per_block=(_OEE_TPB_X, _OEE_TPB_Y)):
            yield s
            init = s
        steps_left -= seg
        if steps_left <= 0:
            break
        current_context = np.random.choice(num_contexts, p=probabilities)
        net = networks[current_context]
        if len(net) == 4:
            nodes, functions, entangled_pairs, _kr = net
        else:
            nodes, functions, entangled_pairs = net
            _kr = KripkeFrame(states=nodes)

# ================= main simulator (unchanged) =================
def simulate_pbn(networks, probabilities, steps=10000, switching=0.5,
                 use_gpu=False, gpu_update='synchronous', pbn_dependent=True,
                 gpu_chunk_size=None, threads_per_block=None, verbose=False):
    if use_gpu and _CUBEWALKERS_AVAILABLE and _all_contexts_pure_boolean(networks):
        try:
            first_nodes = networks[0][0]
            chunk_T = gpu_chunk_size or _estimate_gpu_chunk_T(len(first_nodes))
            return _gpu_state_stream_PBN(networks, probabilities, steps, switching,
                                         gpu_update=gpu_update, pbn_dependent=pbn_dependent,
                                         chunk_T=chunk_T)

        except Exception as e:
            if verbose:
                print(f"[simulate_pbn] GPU path failed: {e}. Falling back to CPU.")
    # CPU baseline
    num_contexts = len(networks)
    current_context = np.random.choice(num_contexts, p=probabilities)
    net = networks[current_context]
    if len(net) == 4:
        nodes, functions, entangled_pairs, kripke_frame = net
    else:
        nodes, functions, entangled_pairs = net
        kripke_frame = KripkeFrame(states=nodes)
    current_state = {node: random.choice([0, 1]) for node in nodes}
    states = [list(current_state.values())]
    for step in range(steps):
        if random.random() < switching:
            current_context = np.random.choice(num_contexts, p=probabilities)
            net = networks[current_context]
            if len(net) == 4:
                nodes, functions, entangled_pairs, kripke_frame = net
            else:
                nodes, functions, entangled_pairs = net
                kripke_frame = KripkeFrame(states=nodes)
        next_state = {}
        context_factor = (step + 1) / steps
        for node in nodes:
            regulators, lookup_table = functions[node]
            input_state = tuple(current_state[reg] for reg in regulators)
            value_entry = lookup_table.get(input_state, random.choice([0, 1]))
            accessible_values = [current_state[acc] for acc in kripke_frame.get_accessible_states(node)
                                 if acc in current_state]
            if isinstance(value_entry, int):
                value = value_entry
            elif isinstance(value_entry, str) and value_entry == "superposed":
                value = collapse_superposition(value_entry)
            elif isinstance(value_entry, str) and value_entry == "possible":
                value = resolve_modal_state(value_entry, context_factor, accessible_values)
            elif isinstance(value_entry, tuple) and value_entry[0] == "necessary":
                value = resolve_modal_state(value_entry, context_factor, accessible_values)
            elif node in entangled_pairs:
                entangled_node = entangled_pairs[node]
                value = current_state[entangled_node]
            else:
                value = resolve_contradiction(input_state)
            next_state[node] = value
        current_state = next_state
        states.append(list(current_state.values()))
    return states

# =================== NEW: exact piecewise KD ===================

def _ctx_is_pure_boolean(ctx):
    if len(ctx) == 4:
        _, funs, ent, _ = ctx
    else:
        _, funs, ent = ctx
    return (not ent) and _is_pure_boolean_functions(funs)

def _cpu_step_boolean(nodes, functions, state_list):
    """Deterministic update for pure-Boolean contexts; state_list is list[int] of length N."""
    sdict = {n: state_list[i] for i, n in enumerate(nodes)}
    out = []
    for n in nodes:
        regs, lut = functions[n]
        key = tuple(sdict[r] for r in regs)
        out.append(int(lut.get(key, 0)))
    return out

def measure_kd_piecewise(networks, probabilities, steps, switching=0.5,
                         use_gpu=True, gpu_update='synchronous', gpu_chunk_size=None):
    """
    Exact KD/V/P for PBNs whenever every context is deterministic (pure Boolean, no entanglement).
    For non-deterministic contexts, falls back to full simulation + extractor (keeps correctness).
    """
    pure = all(_ctx_is_pure_boolean(ctx) for ctx in networks)
    if not pure:
        # Fallback: preserve exactness via your original streaming path.
        from single import extract_attractor_metrics
        states = simulate_pbn(networks, probabilities, steps, switching,
                              use_gpu=use_gpu, gpu_update=gpu_update, gpu_chunk_size=gpu_chunk_size)
        V, P, KD = extract_attractor_metrics(states)
        return V, P, KD

    # Deterministic piecewise exact path
    num_contexts = len(networks)
    current_context = np.random.choice(num_contexts, p=probabilities)
    ctx = networks[current_context]
    nodes = ctx[0]
    functions = ctx[1]
    N = len(nodes)
    state = [random.choice([0, 1]) for _ in range(N)]  # initial system state

    V_sum = 0
    P_sum = 0
    KD_sum = 0

    steps_left = int(steps)
    # GPU availability for this run?
    gpu_ok = bool(use_gpu and _CUBEWALKERS_AVAILABLE)
    chunk_T = gpu_chunk_size or _estimate_gpu_chunk_T(N)

    while steps_left > 0:
        # draw a segment length (geometric stay) capped by remaining steps
        L = steps_left if switching <= 0 else min(int(np.random.geometric(p=switching)), steps_left)

        # ---- run until first repeat OR segment exhausted ----
        seen = {}              # state tuple -> index in segment
        seg_states = [tuple(state)]  # include the starting state at index 0
        seen[seg_states[0]] = 0
        entered = None
        lam = None

        remaining = L
        # we'll stream in small subchunks to allow early stop on GPU
        while remaining > 0 and lam is None:
            sub = min(remaining, max(1, chunk_T if gpu_ok else remaining))
            if gpu_ok:
                # stream sub steps from GPU
                for s_next in _gpu_state_stream_for_context(nodes, functions, seg_states[-1], sub,
                				            gpu_update=gpu_update, is_pbn=False,
                                                            pbn_dependent=True,  # ignored for non-PBN 
                                                            chunk_T=chunk_T,
                                                            threads_per_block=(_OEE_TPB_X, _OEE_TPB_Y)):
                    st = tuple(s_next)
                    seg_states.append(st)
                    if st in seen:
                        entered = seen[st]
                        lam = len(seg_states) - 1 - entered
                        break
                    seen[st] = len(seg_states) - 1
                # if we broke early, discard remainder of this subchunk
                if lam is None:
                    remaining -= sub
                else:
                    # we consumed exactly 1 step that closed the cycle
                    consumed = (len(seg_states) - 1)  # transitions performed in this segment so far
                    remaining = L - consumed
            else:
                # CPU single-step loop (deterministic)
                for _ in range(sub):
                    state = _cpu_step_boolean(nodes, functions, state)
                    st = tuple(state)
                    seg_states.append(st)
                    if st in seen:
                        entered = seen[st]
                        lam = len(seg_states) - 1 - entered
                        break
                    seen[st] = len(seg_states) - 1
                if lam is None:
                    remaining -= sub
                else:
                    consumed = (len(seg_states) - 1)
                    remaining = L - consumed

        # ---- accumulate segment contribution and compute final state ----
        if lam is not None and entered is not None and L > entered:
            dwell = L - entered
            V_sum  += lam
            P_sum  += dwell
            KD_sum += lam * dwell

            # final state at end of segment: advance (remaining) steps along the cycle
            cycle_seq = seg_states[entered: entered + lam]  # ordered cycle
            # we are currently at the repeated state (position 0 in cycle_seq)
            final_idx = remaining % lam
            final_state = list(cycle_seq[final_idx])
        else:
            # no cycle reached inside this segment
            final_state = list(seg_states[-1])

        # move to next segment
        state = final_state
        steps_left -= L
        if steps_left <= 0:
            break
        current_context = np.random.choice(num_contexts, p=probabilities)
        ctx = networks[current_context]
        nodes = ctx[0]; functions = ctx[1]

    return V_sum, P_sum, KD_sum

def simulate_and_measure(idx, num_nodes, avg_connectivity, topology, bias, steps, transformations, num_contexts, context_probabilities):
    """Generates a Boolean Network, applies transformations, and measures open-endedness with PBN modeling."""
    nodes, functions = generate_boolean_network(num_nodes, avg_connectivity, topology, bias)
    entangled_pairs = {}
    
    # Store all transformations as different contexts in a PBN
    pbn_networks = []
    results = {}
    
    # Original Boolean Network
    pbn_networks.append((nodes, functions, entangled_pairs))
    bn_states = simulate_pbn([(nodes, functions, entangled_pairs)], [1.0], steps)
    results["Original"] = measure_open_endedness(bn_states)
    
    # Apply each individual transformation and store it as a new context
    transformed_networks = {"Original": (nodes, functions, entangled_pairs)}
    for name, transform in transformations.items():
        transformed_output = transform((nodes, functions, entangled_pairs))
        
        # Handle transformations that return 3 or 4 elements
        if len(transformed_output) == 4:
            transformed_nodes, transformed_functions, transformed_entangled_pairs, transformed_kripke_frame = transformed_output
        else:
            transformed_nodes, transformed_functions, transformed_entangled_pairs = transformed_output
            transformed_kripke_frame = KripkeFrame(states=transformed_nodes)  # Ensure a Kripke frame exists
        
        transformed_networks[name] = (transformed_nodes, transformed_functions, transformed_entangled_pairs, transformed_kripke_frame)
        pbn_networks.append((transformed_nodes, transformed_functions, transformed_entangled_pairs, transformed_kripke_frame))
        
        transformed_states = simulate_pbn([(transformed_nodes, transformed_functions, transformed_entangled_pairs, transformed_kripke_frame)], [1.0], steps)
        results[name] = measure_open_endedness(transformed_states)
    
    # Apply valid transformation combinations
    combination_orders = [
        ("Paraconsistent", "Modal"),
        ("Paraconsistent", "Quantum"),
        ("Paraconsistent", "Dynamic"),
        ("Modal", "Quantum"),
        ("Modal", "Dynamic"),
    ]
    
    for combo in combination_orders:
        transformed_network = (nodes, functions, entangled_pairs)
        
        for transformation in combo:
            transformed_network = transformations[transformation](transformed_network)
        
        if len(transformed_network) == 4:
            transformed_nodes, transformed_functions, transformed_entangled_pairs, transformed_kripke_frame = transformed_network
        else:
            transformed_nodes, transformed_functions, transformed_entangled_pairs = transformed_network
            transformed_kripke_frame = KripkeFrame(states=transformed_nodes)

        transformed_networks[" + ".join(combo)] = transformed_network
        pbn_networks.append(transformed_network)
        
        transformed_states = simulate_pbn([transformed_network], [1.0], steps)
        results[" + ".join(combo)] = measure_open_endedness(transformed_states)
    
    # Apply PBN transformation with all valid transformations
    for key in list(results.keys()):
        transformed_network = transformed_networks[key]
        
        # Build the PBN using the specified number of contexts and probability distribution
        pbn_networks = random.sample(list(transformed_networks.values()), num_contexts)
        transformed_states = simulate_pbn(pbn_networks, context_probabilities, steps)
        
        results[f"PBN({key})"] = measure_open_endedness(transformed_states)
    
    return results

# Main exploration loop
'''
def main():
    num_networks = 10  # Adjust as needed
    num_nodes = 100  
    avg_connectivity = 2  
    topology = "Poisson"
    bias = 0.5
    steps = 1000
    num_contexts = 10  # Define the number of contexts
    context_probabilities = np.ones(num_contexts) / num_contexts  # Uniform distribution

    transformations = {
        "Paraconsistent": apply_paraconsistent_logic,
        "Modal": apply_modal_logic,
        "Quantum": apply_quantum_logic,
        "Dynamic": apply_dynamic_logic,
    }

    # Parallel execution
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results_iterator = pool.imap(
            partial(simulate_and_measure, num_nodes=num_nodes, avg_connectivity=avg_connectivity, 
                    topology=topology, bias=bias, steps=steps, transformations=transformations,
                    num_contexts=num_contexts, context_probabilities=context_probabilities),
            range(num_networks)
        )

        all_results = list(tqdm(results_iterator, total=num_networks, desc="Simulating Networks"))

    # Aggregate results
    avg_results = {key: 0 for key in all_results[0].keys()}
    for result in all_results:
        for key, value in result.items():
            avg_results[key] += value

    for key in avg_results:
        avg_results[key] /= num_networks  # Compute averages

    print("Average Open-Endedness Values:")
    for key, value in avg_results.items():
        print(f"{key}: {value}")

    return avg_results
'''

if __name__ == "__main__":
    main()
