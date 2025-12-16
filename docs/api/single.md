# single.py

## `compute_total_variety(attractors)`

## `compute_total_persistence(attractors)`

## `compute_total_basin_length(attractors)`

## `extract_attractor_metrics(network_states)`

Returns (V, P, KD). Works with a NumPy array (T,N) *or* any iterable of states.

## `simulate_rbn_and_measure_metrics(num_nodes, avg_connectivity, topology, bias, steps)`

## `simulate_rbn_and_measure_metrics_cycle_tail(num_nodes, avg_connectivity, topology, bias, steps, use_gpu=False)`

Exact long-T OEE for a single deterministic BN by stopping once the first cycle is found,
    then filling the tail analytically: V=lambda, P=T-mu, KD=lambda*(T-mu).

## `process_networks(avg_connectivity, num_networks, num_nodes, topology, bias, steps)`

## `process_networks(avg_connectivity, num_networks, num_nodes, topology, bias, steps)`

For large T (GPU mode), avoid spawning many processes (single GPU contention).
    For small T, keep your original CPU parallel loop.

## `main()`
