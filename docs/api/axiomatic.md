# axiomatic.py

## class `KripkeFrame`

## `__init__(self, states)`

## `add_accessibility(self, from_state, to_state)`

## `get_accessible_states(self, state)`

## `kolmogorov_complexity(state)`

## `sophistication(network_states)`

## `coarse_sophistication(network_states)`

## `busy_beaver_logical_depth(network_states)`

## `generate_boolean_network(num_nodes, avg_connectivity, topology, bias)`

## `apply_paraconsistent_logic(network, contradiction_prob=0.3)`

## `propagate_paraconsistent_state(value)`

## `resolve_contradiction(inputs)`

## `apply_modal_logic(network, accessibility_degree=None, p_possible=0.3, p_necessary=0.3,
                      nec_strength_range=(0.7, 1.0))`

## `resolve_modal_state(value, context_factor, accessible_values)`

## `apply_quantum_logic(network, superposition_prob=0.3, max_entangled_pairs=5)`

## `collapse_superposition(value)`

## `mutate_rule(value)`

## `apply_dynamic_logic(network, mutation_prob=0.5)`

## `_is_pure_boolean_functions(functions)`

## `_all_contexts_pure_boolean(networks)`

## `_build_regulators_and_lut(nodes, functions)`

## `_estimate_gpu_chunk_T(N, bytes_per_val=1, max_frac=0.20, hard_cap=1_000_000)`

## `_choose_cw_mask(update: str, is_pbn: bool, pbn_dependent: bool)`

update ∈ {"synchronous","asynchronous","asynchronous_set"}
    is_pbn: use PBN-aware mask variants
    pbn_dependent: use *dependent* variants that share a single PBN value across nodes (per walker/step)

## `_gpu_state_stream_for_context(nodes, functions, initial_state_np, steps,
                                  gpu_update='synchronous', is_pbn=False, pbn_dependent=True,
                                  chunk_T=None, threads_per_block=None)`

## `_gpu_state_stream_PBN(networks, probabilities, steps, switching,
                          gpu_update='synchronous', pbn_dependent=True, chunk_T=None)`

## `simulate_pbn(networks, probabilities, steps=10000, switching=0.5,
                 use_gpu=False, gpu_update='synchronous', pbn_dependent=True,
                 gpu_chunk_size=None, threads_per_block=None, verbose=False)`

## `_ctx_is_pure_boolean(ctx)`

## `_cpu_step_boolean(nodes, functions, state_list)`

Deterministic update for pure-Boolean contexts; state_list is list[int] of length N.

## `measure_kd_piecewise(networks, probabilities, steps, switching=0.5,
                         use_gpu=True, gpu_update='synchronous', gpu_chunk_size=None)`

Exact KD/V/P for PBNs whenever every context is deterministic (pure Boolean, no entanglement).
    For non-deterministic contexts, falls back to full simulation + extractor (keeps correctness).

## `simulate_and_measure(idx, num_nodes, avg_connectivity, topology, bias, steps, transformations, num_contexts, context_probabilities)`

Generates a Boolean Network, applies transformations, and measures open-endedness with PBN modeling.

## `main()`
