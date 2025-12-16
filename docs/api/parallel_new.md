# parallel_new.py

## `_invoke_sample_once(args)`

## `_ci_halfwidth(samples, z=1.96)`

## `_run_until_ci(sample_once_fn, fixed_kwargs, batch=50, max_n=1000, delta=0.02, use_pool=True)`

Repeatedly call sample_once_fn(**fixed_kwargs) in small batches until
    the 95% CI half-width <= delta, or until max_n samples are drawn.
    Returns (mean, n_used).

## `_env_steps(default_steps)`

## `_env_delta(default=0.02)`

## `_env_batch(default=50)`

## `_env_maxnets(default=1000)`

## `dynamic_sample_once(K, num_nodes, topology, bias, steps, mutation_prob)`

## `paraconsistent_sample_once(K, num_nodes, topology, bias, steps, contradiction_prob)`

## `modal_sample_once(K, num_nodes, topology, bias, steps, accessibility_degree, p_possible, p_necessary)`

## `quantum_sample_once(K, num_nodes, topology, bias, steps, superposition_prob, max_entangled)`

## `pbn_sample_once(K, num_nodes, topology, bias, steps, switch_prob, num_contexts)`

Build a PBN with `num_contexts` pure-Boolean contexts and compute one OEE sample.
    Uses measure_kd_piecewise (exact when all contexts are deterministic).

## `compute_open_endedness(states_or_VPK, T)`

## `process_network(args)`

## `run_simulation(avg_connectivity, num_networks=1000, num_nodes=100, steps=10000)`

## `parallel_run(args)`

## `quantum_run(args)`

## `dynamic_run(args)`

## `paraconsistent_run(args)`

args = (K, num_networks, num_nodes, topology, bias, steps, contradiction_prob)

## `modal_run(args)`

args = (K, num_networks, num_nodes, topology, bias, steps,
            accessibility_degree, p_possible, p_necessary)

## `main()`

## `main_quantum_logic()`

## `main_dynamic_logic()`

## `main_paraconsistent_logic()`

## `main_modal_logic()`

## `quick_sanity_check()`

Fast surrogate for the 3.5-hour run.
    ─────────────────────────────────────────
    •  K grid       : 3 values that hit ordered / critical / chaotic
    •  networks     : 40 (instead of 1 000)
    •  contexts     : 5  (same as main experiment)
    •  run length   : 5 000 steps
    •  aggregation  : median  (more robust for tiny sample)
    Produces a figure and prints medians so you can see immediately
    whether the PBN belly is present before committing to the full job.
