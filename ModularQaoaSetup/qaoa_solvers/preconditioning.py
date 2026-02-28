from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Sequence

import networkx as nx
import numpy as np

from .baselines import apply_greedy_block_update, build_standard_qaoa_circuit, optimize_standard_qaoa
from .hyperparameters import StandardQAOAOptimizerConfig
from ..error_modeling.models import simulate_circuit_probabilities_with_error_model
from ..graph_ops import (
    burer_monteiro_refine,
    greedy_refine,
    initial_assignment,
    ising_energy,
    simulated_annealing_refine,
    weighted_cut_value,
)
from .core import EPS, SolveResult, basis_spins, chunked_exact_cut
from ..partitioning_methods.strategies import PartitionSchedule, build_partition_schedule
from ..pipeline import run_block_coordinate_descent
from ..resource_accounting import record_checkpoint, record_quantum_block_update


if TYPE_CHECKING:
    from .modular import ModularSolveConfig


@dataclass(frozen=True)
class PreconditionedGraphBuild:
    graph: nx.Graph
    schedule: PartitionSchedule
    candidate_pairs: int
    correlated_pairs: int
    light_cone_evaluations: int
    max_light_cone_size: int


def _standard_qaoa_gate_counts(pair_weights: np.ndarray, depth: int) -> tuple[int, int]:
    pair_terms = int(np.count_nonzero(np.triu(np.abs(pair_weights) > EPS, k=1)))
    one_qubit_gate_count = pair_weights.shape[0] + depth * pair_weights.shape[0]
    two_qubit_gate_count = depth * pair_terms
    return one_qubit_gate_count, two_qubit_gate_count


def _zero_field_pair_weights(
    graph: nx.Graph,
    nodes: Sequence[int],
) -> tuple[np.ndarray, Dict[int, int]]:
    ordered_nodes = tuple(sorted(int(node) for node in nodes))
    size = len(ordered_nodes)
    pair_weights = np.zeros((size, size), dtype=float)
    node_to_index = {node: index for index, node in enumerate(ordered_nodes)}
    subgraph = graph.subgraph(ordered_nodes)

    for u, v, data in subgraph.edges(data=True):
        i = node_to_index[int(u)]
        j = node_to_index[int(v)]
        weight = float(data["weight"])
        pair_weights[i, j] = weight
        pair_weights[j, i] = weight

    return pair_weights, node_to_index


def _light_cone_nodes(
    graph: nx.Graph,
    seed_nodes: Sequence[int],
    depth: int,
    max_nodes: int,
) -> tuple[int, ...]:
    ordered_seed_nodes = tuple(sorted(set(int(node) for node in seed_nodes)))
    if not ordered_seed_nodes:
        return ()
    cutoff = max(1, int(depth))
    distances: Dict[int, int] = {}
    for seed in ordered_seed_nodes:
        for node, distance in nx.single_source_shortest_path_length(graph, seed, cutoff=cutoff).items():
            previous = distances.get(int(node))
            local_distance = int(distance)
            if previous is None or local_distance < previous:
                distances[int(node)] = local_distance

    candidates = set(distances)
    if len(candidates) <= max_nodes:
        return tuple(sorted(int(node) for node in candidates))

    endpoints = set(ordered_seed_nodes)
    ordered = sorted(
        candidates,
        key=lambda node: (
            0 if node in endpoints else 1,
            distances.get(int(node), cutoff + 1),
            int(node),
        ),
    )

    keep = set(ordered[: max(2, max_nodes)])
    keep.update(endpoints)
    return tuple(sorted(int(node) for node in keep))


def _seed_blocks(schedule: PartitionSchedule) -> tuple[tuple[int, ...], ...]:
    blocks: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for group in (schedule.region_blocks, schedule.boundary_blocks):
        for block in group:
            normalized = tuple(sorted(set(int(node) for node in block)))
            if len(normalized) < 2 or normalized in seen:
                continue
            seen.add(normalized)
            blocks.append(normalized)
    return tuple(blocks)


def build_standard_qaoa_preconditioned_graph(
    graph: nx.Graph,
    *,
    depth: int,
    restarts: int,
    maxiter: int,
    seed: int,
    schedule: PartitionSchedule | None = None,
    max_block_size: int = 16,
    partition_strategy: str = "recursive_spectral_kl",
    max_light_cone_size: int | None = None,
    min_abs_weight: float = 1e-6,
    error_model: str = "ideal",
    error_model_specs_path: str = "Ankaa-3_device_specs.csv",
    error_model_layout_seed: int = 0,
    optimizer_config: StandardQAOAOptimizerConfig | None = None,
) -> PreconditionedGraphBuild:
    active_schedule = schedule
    if active_schedule is None:
        active_schedule = build_partition_schedule(
            graph,
            max_block_size=max_block_size,
            seed=seed,
            strategy=partition_strategy,
        )

    seed_blocks = _seed_blocks(active_schedule)
    light_cone_limit = max(
        2,
        max((len(block) for block in seed_blocks), default=2),
        int(max_light_cone_size)
        if max_light_cone_size is not None
        else max(8, min(24, max_block_size + 2 * max(1, depth))),
    )
    cached_correlations: Dict[tuple[int, ...], tuple[Dict[int, int], np.ndarray]] = {}
    weight_totals: Dict[tuple[int, int], float] = {}
    weight_counts: Dict[tuple[int, int], int] = {}
    candidate_pairs: set[tuple[int, int]] = set()
    max_observed_light_cone = 0
    light_cone_evaluations = 0

    for block_index, block_nodes in enumerate(seed_blocks):
        cone_nodes = _light_cone_nodes(
            graph,
            seed_nodes=block_nodes,
            depth=depth,
            max_nodes=light_cone_limit,
        )
        if len(cone_nodes) < 2:
            continue

        max_observed_light_cone = max(max_observed_light_cone, len(cone_nodes))
        cached = cached_correlations.get(cone_nodes)
        if cached is None:
            pair_weights, node_to_index = _zero_field_pair_weights(graph, cone_nodes)
            fields = np.zeros(len(cone_nodes), dtype=float)
            record_quantum_block_update(len(cone_nodes))
            local_seed = seed + 1009 * (block_index + 1)
            params, _ = optimize_standard_qaoa(
                pair_weights,
                fields,
                depth=depth,
                restarts=restarts,
                maxiter=maxiter,
                seed=local_seed,
                error_model=error_model,
                error_model_specs_path=error_model_specs_path,
                error_model_layout_seed=error_model_layout_seed,
                optimizer_config=optimizer_config,
            )
            circuit = build_standard_qaoa_circuit(
                pair_weights,
                fields,
                params[:depth],
                params[depth:],
            )
            one_qubit_gate_count, two_qubit_gate_count = _standard_qaoa_gate_counts(pair_weights, depth)
            probabilities, _ = simulate_circuit_probabilities_with_error_model(
                circuit,
                num_qubits=len(cone_nodes),
                one_qubit_gate_count=one_qubit_gate_count,
                two_qubit_gate_count=two_qubit_gate_count,
                measurement_qubits=len(cone_nodes),
                error_model=error_model,
                specs_path=error_model_specs_path,
                layout_seed=error_model_layout_seed + block_index,
            )
            spins = basis_spins(len(cone_nodes))
            correlations = spins.T @ (spins * probabilities[:, None])
            cached = (node_to_index, correlations)
            cached_correlations[cone_nodes] = cached
            light_cone_evaluations += 1

        node_to_index, correlations = cached
        for left_index in range(len(block_nodes)):
            u = int(block_nodes[left_index])
            for right_index in range(left_index + 1, len(block_nodes)):
                v = int(block_nodes[right_index])
                pair = (u, v) if u < v else (v, u)
                candidate_pairs.add(pair)
                correlation = float(correlations[node_to_index[u], node_to_index[v]])
                weight = -correlation
                if abs(weight) <= max(min_abs_weight, EPS):
                    continue
                weight_totals[pair] = weight_totals.get(pair, 0.0) + weight
                weight_counts[pair] = weight_counts.get(pair, 0) + 1

    preconditioned_graph = nx.Graph()
    preconditioned_graph.add_nodes_from(int(node) for node in graph.nodes())
    for (u, v), total_weight in weight_totals.items():
        count = max(1, weight_counts[(u, v)])
        average_weight = total_weight / count
        if abs(average_weight) > max(min_abs_weight, EPS):
            preconditioned_graph.add_edge(int(u), int(v), weight=float(average_weight))

    return PreconditionedGraphBuild(
        graph=preconditioned_graph,
        schedule=active_schedule,
        candidate_pairs=len(candidate_pairs),
        correlated_pairs=preconditioned_graph.number_of_edges(),
        light_cone_evaluations=light_cone_evaluations,
        max_light_cone_size=max_observed_light_cone,
    )


def solve_graph_quantum_preconditioned(
    graph: nx.Graph,
    name: str,
    config: "ModularSolveConfig",
) -> SolveResult:
    build = build_standard_qaoa_preconditioned_graph(
        graph,
        depth=config.depth,
        restarts=config.restarts,
        maxiter=config.maxiter,
        seed=config.seed,
        max_block_size=config.max_block_size,
        partition_strategy=config.partition_strategy,
        max_light_cone_size=config.max_light_cone_size,
        min_abs_weight=config.preconditioner_min_abs_weight,
        error_model=config.error_model,
        error_model_specs_path=config.error_model_specs_path,
        error_model_layout_seed=config.error_model_layout_seed,
        optimizer_config=config.standard_qaoa_optimizer,
    )
    schedule = build.schedule
    working_graph = build.graph

    assignment = initial_assignment(working_graph, config.seed)
    backend = config.preconditioning_backend.strip().lower()
    if backend not in {"simulated_annealing", "burer_monteiro", "greedy"}:
        raise ValueError(f"Unknown preconditioning backend: {config.preconditioning_backend}")
    if backend == "simulated_annealing":
        assignment = simulated_annealing_refine(
            working_graph,
            assignment,
            blocks=schedule.region_blocks,
            temperatures=max(2, config.sa_temperatures // 2),
            sweeps_per_temperature=1,
            initial_temperature=config.sa_initial_temperature,
            final_temperature=config.sa_final_temperature,
            seed=config.seed,
        )
    elif backend == "burer_monteiro":
        assignment = burer_monteiro_refine(
            working_graph,
            assignment,
            rank=config.bm_rank,
            steps=max(8, config.bm_steps // 2),
            learning_rate=config.bm_learning_rate,
            rounding_trials=max(8, config.bm_rounding_trials // 2),
            seed=config.seed,
        )
    else:
        assignment = greedy_refine(working_graph, assignment, seed=config.seed)
    record_checkpoint(weighted_cut_value(graph, assignment), label="initial")

    if backend == "simulated_annealing":

        def region_updater(
            run_graph: nx.Graph,
            block: Sequence[int],
            current_assignment: Dict[int, int],
            local_seed: int,
        ) -> None:
            simulated_annealing_refine(
                run_graph,
                current_assignment,
                blocks=(block,),
                temperatures=config.sa_temperatures,
                sweeps_per_temperature=config.sa_sweeps_per_temperature,
                initial_temperature=config.sa_initial_temperature,
                final_temperature=config.sa_final_temperature,
                seed=local_seed,
            )

        boundary_updater = region_updater
    elif backend == "burer_monteiro":

        def region_updater(
            run_graph: nx.Graph,
            block: Sequence[int],
            current_assignment: Dict[int, int],
            local_seed: int,
        ) -> None:
            burer_monteiro_refine(
                run_graph,
                current_assignment,
                blocks=(block,),
                rank=config.bm_rank,
                steps=config.bm_steps,
                learning_rate=config.bm_learning_rate,
                rounding_trials=config.bm_rounding_trials,
                seed=local_seed,
            )

        boundary_updater = region_updater
    else:

        def region_updater(
            run_graph: nx.Graph,
            block: Sequence[int],
            current_assignment: Dict[int, int],
            local_seed: int,
        ) -> None:
            apply_greedy_block_update(run_graph, block, current_assignment, seed=local_seed)

        boundary_updater = region_updater

    pipeline_result = run_block_coordinate_descent(
        working_graph,
        assignment,
        region_blocks=schedule.region_blocks,
        boundary_blocks=schedule.boundary_blocks,
        rounds=config.rounds,
        seed=config.seed,
        refine_assignment=lambda run_graph, current, refine_seed: greedy_refine(
            run_graph,
            current,
            seed=refine_seed,
        ),
        objective_value=lambda _run_graph, current: weighted_cut_value(graph, current),
        region_updater=region_updater,
        boundary_updater=boundary_updater,
        shuffle_region_blocks=schedule.layout is None,
        shuffle_boundary_blocks=False,
    )

    assignment = pipeline_result.assignment
    if backend == "simulated_annealing":
        polish_blocks = tuple(schedule.region_blocks) + tuple(schedule.boundary_blocks)
        assignment = simulated_annealing_refine(
            graph,
            assignment,
            blocks=polish_blocks if polish_blocks else None,
            temperatures=max(4, config.sa_temperatures // 2),
            sweeps_per_temperature=max(1, config.sa_sweeps_per_temperature),
            initial_temperature=config.sa_initial_temperature,
            final_temperature=config.sa_final_temperature,
            seed=config.seed + 9000,
        )
        record_checkpoint(weighted_cut_value(graph, assignment), label="original_sa_polish")
    elif backend == "burer_monteiro":
        assignment = burer_monteiro_refine(
            graph,
            assignment,
            rank=config.bm_rank,
            steps=max(12, config.bm_steps),
            learning_rate=config.bm_learning_rate,
            rounding_trials=max(12, config.bm_rounding_trials),
            seed=config.seed + 9000,
        )
        record_checkpoint(weighted_cut_value(graph, assignment), label="original_bm_polish")

    assignment = greedy_refine(graph, assignment, seed=config.seed + 10000)
    record_checkpoint(weighted_cut_value(graph, assignment), label="original_greedy_polish")
    exact_cut_value = None
    if graph.number_of_nodes() <= config.exact_threshold:
        exact_cut_value, exact_assignment = chunked_exact_cut(graph)
        if config.allow_exact_postcheck and exact_cut_value > weighted_cut_value(graph, assignment):
            assignment = exact_assignment

    blocks = list(schedule.layout.regions) if schedule.layout is not None else list(schedule.region_blocks)
    coarse_nodes = 0 if schedule.layout is None else schedule.layout.coarse_graph.number_of_nodes()
    boundary_nodes = 0 if schedule.layout is None else len(schedule.layout.boundary_nodes)

    return SolveResult(
        name=name,
        graph=graph,
        assignment=assignment,
        blocks=blocks,
        rounds_run=pipeline_result.rounds_run,
        cut_value=weighted_cut_value(graph, assignment),
        ising_energy=ising_energy(graph, assignment),
        exact_cut_value=exact_cut_value,
        coarse_nodes=coarse_nodes,
        boundary_nodes=boundary_nodes,
        boundary_blocks=len(schedule.boundary_blocks),
        strategy=(
            f"{config.partition_strategy}:"
            f"standard_qaoa_preconditioned_{backend}"
            f"[pairs={build.correlated_pairs},cones={build.light_cone_evaluations}]"
        ),
    )
