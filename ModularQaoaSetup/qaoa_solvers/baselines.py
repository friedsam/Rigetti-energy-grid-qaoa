from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms.approximation.maxcut import one_exchange, randomized_partitioning
from qiskit.circuit.library import QAOAAnsatz
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.optimize import minimize

from ..error_modeling.models import simulate_circuit_probabilities_with_error_model
from ..graph_ops import greedy_refine, ising_energy, weighted_cut_value
from ..partitioning_methods.strategies import boundary_blocks_from_layout, build_partition_schedule, build_region_layout
from ..pipeline import run_block_coordinate_descent
from ..resource_accounting import (
    record_classical_greedy_block,
    record_quantum_block_update,
    record_quantum_decode,
    record_quantum_statevector_evaluation,
)
from .core import (
    EPS,
    SolveResult,
    assign_regions_from_coarse,
    basis_spins,
    build_local_problem,
    build_warm_start_qaoa,
    chunked_exact_cut,
    discover_parquet_files,
    exact_block_assignment,
    load_weighted_graph,
    optimize_warm_start_qaoa,
    solve_graph,
)
from .hyperparameters import LibraryQAOAOptimizerConfig, StandardQAOAOptimizerConfig, WarmStartQAOAOptimizerConfig


@dataclass
class BaselineResult:
    graph_name: str
    method: str
    family: str
    assignment: Dict[int, int]
    cut_value: float
    ising_energy: float
    exact_cut_value: float | None = None
    metadata: Dict[str, object] | None = None

    def to_dict(self) -> Dict[str, object]:
        approximation_ratio = None
        if self.exact_cut_value is not None and self.exact_cut_value > 0.0:
            approximation_ratio = self.cut_value / self.exact_cut_value
        return {
            "graph_name": self.graph_name,
            "method": self.method,
            "family": self.family,
            "cut_value": self.cut_value,
            "ising_energy": self.ising_energy,
            "exact_cut_value": self.exact_cut_value,
            "approximation_ratio": approximation_ratio,
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class ComparisonSettings:
    profile: str
    random_search_trials: int
    networkx_restarts: int
    include_quantum: bool
    standard_qaoa_depth: int
    standard_qaoa_max_block_size: int
    standard_qaoa_rounds: int
    standard_qaoa_restarts: int
    standard_qaoa_maxiter: int
    warm_start_qaoa_depth: int
    warm_start_qaoa_max_block_size: int
    warm_start_qaoa_rounds: int
    warm_start_qaoa_restarts: int
    warm_start_qaoa_maxiter: int
    include_hybrid: bool
    hybrid_depth: int
    hybrid_max_block_size: int
    hybrid_rounds: int
    hybrid_restarts: int
    hybrid_maxiter: int
    networkx_partitioned_max_region_size: int
    networkx_partitioned_rounds: int


def get_comparison_settings(profile: str = "balanced") -> ComparisonSettings:
    normalized = profile.strip().lower()
    if normalized == "balanced":
        return ComparisonSettings(
            profile="balanced",
            random_search_trials=128,
            networkx_restarts=3,
            include_quantum=True,
            standard_qaoa_depth=1,
            standard_qaoa_max_block_size=10,
            standard_qaoa_rounds=1,
            standard_qaoa_restarts=1,
            standard_qaoa_maxiter=10,
            warm_start_qaoa_depth=1,
            warm_start_qaoa_max_block_size=10,
            warm_start_qaoa_rounds=1,
            warm_start_qaoa_restarts=1,
            warm_start_qaoa_maxiter=10,
            include_hybrid=True,
            hybrid_depth=1,
            hybrid_max_block_size=16,
            hybrid_rounds=3,
            hybrid_restarts=3,
            hybrid_maxiter=35,
            networkx_partitioned_max_region_size=16,
            networkx_partitioned_rounds=2,
        )
    if normalized == "classical_full":
        return ComparisonSettings(
            profile="classical_full",
            random_search_trials=4096,
            networkx_restarts=8,
            include_quantum=False,
            standard_qaoa_depth=1,
            standard_qaoa_max_block_size=10,
            standard_qaoa_rounds=1,
            standard_qaoa_restarts=1,
            standard_qaoa_maxiter=10,
            warm_start_qaoa_depth=1,
            warm_start_qaoa_max_block_size=10,
            warm_start_qaoa_rounds=1,
            warm_start_qaoa_restarts=1,
            warm_start_qaoa_maxiter=10,
            include_hybrid=True,
            hybrid_depth=1,
            hybrid_max_block_size=16,
            hybrid_rounds=3,
            hybrid_restarts=3,
            hybrid_maxiter=35,
            networkx_partitioned_max_region_size=16,
            networkx_partitioned_rounds=2,
        )
    raise ValueError(f"Unknown comparison profile: {profile}")


def random_assignment(graph: nx.Graph, seed: int) -> Dict[int, int]:
    rng = random.Random(seed)
    return {node: (1 if rng.random() < 0.5 else -1) for node in graph.nodes()}


def assignment_from_partition(graph: nx.Graph, partition: Tuple[Sequence[int], Sequence[int]]) -> Dict[int, int]:
    left, right = partition
    left = set(left)
    right = set(right)
    assignment = {node: (1 if node in left else -1) for node in graph.nodes()}
    if not right:
        for index, node in enumerate(sorted(graph.nodes())):
            assignment[node] = 1 if index % 2 == 0 else -1
    return assignment


def exact_cut_for_graph(graph: nx.Graph, threshold: int) -> float | None:
    if graph.number_of_nodes() > threshold:
        return None
    exact_cut, _ = chunked_exact_cut(graph)
    return exact_cut


def random_search_baseline(
    graph: nx.Graph,
    graph_name: str,
    trials: int = 128,
    seed: int = 0,
    exact_cut_value: float | None = None,
) -> BaselineResult:
    best_assignment = None
    best_cut = -math.inf
    for trial in range(max(1, trials)):
        assignment = random_assignment(graph, seed + trial)
        cut_value = weighted_cut_value(graph, assignment)
        if cut_value > best_cut:
            best_cut = cut_value
            best_assignment = assignment

    if best_assignment is None:
        raise RuntimeError("Random search baseline failed to generate an assignment.")

    return BaselineResult(
        graph_name=graph_name,
        method="Random Search",
        family="classical",
        assignment=best_assignment,
        cut_value=best_cut,
        ising_energy=ising_energy(graph, best_assignment),
        exact_cut_value=exact_cut_value,
        metadata={"trials": trials},
    )


def one_exchange_baseline(
    graph: nx.Graph,
    graph_name: str,
    seed: int = 0,
    exact_cut_value: float | None = None,
) -> BaselineResult:
    try:
        _, partition = one_exchange(graph, weight="weight", seed=seed)
        assignment = assignment_from_partition(graph, partition)
    except Exception:
        assignment = random_assignment(graph, seed)

    assignment = greedy_refine(graph, assignment, seed=seed)
    return BaselineResult(
        graph_name=graph_name,
        method="One-Exchange + Greedy",
        family="classical",
        assignment=assignment,
        cut_value=weighted_cut_value(graph, assignment),
        ising_energy=ising_energy(graph, assignment),
        exact_cut_value=exact_cut_value,
        metadata={"seed": seed},
    )


def networkx_maxcut_baseline(
    graph: nx.Graph,
    graph_name: str,
    restarts: int = 4,
    seed: int = 0,
    exact_cut_value: float | None = None,
) -> BaselineResult:
    best_cut, best_partition = one_exchange(graph, weight="weight", seed=seed)

    for restart in range(max(0, restarts)):
        _, partition = randomized_partitioning(graph, seed=seed + restart, weight="weight")
        candidate_cut, candidate_partition = one_exchange(
            graph,
            initial_cut=partition[0],
            weight="weight",
            seed=seed + restart,
        )
        if candidate_cut > best_cut:
            best_cut = candidate_cut
            best_partition = candidate_partition

    assignment = assignment_from_partition(graph, best_partition)
    return BaselineResult(
        graph_name=graph_name,
        method="NetworkX MaxCut",
        family="classical",
        assignment=assignment,
        cut_value=weighted_cut_value(graph, assignment),
        ising_energy=ising_energy(graph, assignment),
        exact_cut_value=exact_cut_value,
        metadata={"restarts": restarts},
    )


def apply_networkx_block_update(
    graph: nx.Graph,
    block_nodes: Sequence[int],
    assignment: Dict[int, int],
    restarts: int,
    seed: int,
) -> None:
    block_nodes = tuple(sorted(int(node) for node in block_nodes))
    if not block_nodes:
        return
    if len(block_nodes) == 1:
        return

    subgraph = graph.subgraph(block_nodes).copy()
    initial_cut = {node for node in block_nodes if assignment[node] > 0}
    best_value, best_partition = one_exchange(
        subgraph,
        initial_cut=initial_cut,
        weight="weight",
        seed=seed,
    )

    for restart in range(max(0, restarts)):
        _, partition = randomized_partitioning(
            subgraph,
            seed=seed + restart,
            weight="weight",
        )
        candidate_value, candidate_partition = one_exchange(
            subgraph,
            initial_cut=partition[0],
            weight="weight",
            seed=seed + restart,
        )
        if candidate_value > best_value:
            best_value = candidate_value
            best_partition = candidate_partition

    left, right = best_partition
    left = set(left)
    same_score = sum(1 for node in block_nodes if ((node in left) == (assignment[node] > 0)))
    flip_score = len(block_nodes) - same_score
    use_left_as_positive = same_score >= flip_score

    for node in block_nodes:
        in_left = node in left
        if use_left_as_positive:
            assignment[node] = 1 if in_left else -1
        else:
            assignment[node] = -1 if in_left else 1


def partitioned_networkx_maxcut_baseline(
    graph: nx.Graph,
    graph_name: str,
    restarts: int = 4,
    max_region_size: int = 16,
    rounds: int = 2,
    seed: int = 0,
    exact_cut_value: float | None = None,
) -> BaselineResult:
    layout = build_region_layout(graph, max_region_size=max_region_size, seed=seed)
    coarse_result = networkx_maxcut_baseline(
        layout.coarse_graph,
        graph_name=f"{graph_name}::coarse",
        restarts=restarts,
        seed=seed,
        exact_cut_value=None,
    )
    assignment = assign_regions_from_coarse(layout.regions, coarse_result.assignment)

    for round_index in range(max(1, rounds)):
        for region_index, region in enumerate(layout.regions):
            apply_networkx_block_update(
                graph=graph,
                block_nodes=region,
                assignment=assignment,
                restarts=restarts,
                seed=seed + round_index * 1009 + region_index,
            )

        boundary_blocks = boundary_blocks_from_layout(
            graph,
            layout=layout,
            max_block_size=max_region_size,
            seed=seed + 5000 + round_index,
        )
        for block_index, block in enumerate(boundary_blocks):
            apply_networkx_block_update(
                graph=graph,
                block_nodes=block,
                assignment=assignment,
                restarts=restarts,
                seed=seed + 7000 + round_index * 1009 + block_index,
            )

        assignment = greedy_refine(graph, assignment, seed=seed + round_index)

    return BaselineResult(
        graph_name=graph_name,
        method="NetworkX MaxCut (Partitioned)",
        family="classical",
        assignment=assignment,
        cut_value=weighted_cut_value(graph, assignment),
        ising_energy=ising_energy(graph, assignment),
        exact_cut_value=exact_cut_value,
        metadata={
            "restarts": restarts,
            "max_region_size": max_region_size,
            "rounds": rounds,
            "coarse_nodes": layout.coarse_graph.number_of_nodes(),
            "boundary_nodes": len(layout.boundary_nodes),
        },
    )


def build_standard_qaoa_circuit(
    pair_weights: np.ndarray,
    fields: np.ndarray,
    gammas: np.ndarray,
    betas: np.ndarray,
) -> QuantumCircuit:
    num_qubits = pair_weights.shape[0]
    circuit = QuantumCircuit(num_qubits)
    for qubit in range(num_qubits):
        circuit.h(qubit)

    for layer in range(len(gammas)):
        gamma = float(gammas[layer])
        beta = float(betas[layer])

        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                weight = float(pair_weights[i, j])
                if abs(weight) > EPS:
                    circuit.rzz(-2.0 * gamma * weight, i, j)

        for qubit, field in enumerate(fields):
            field = float(field)
            if abs(field) > EPS:
                circuit.rz(-2.0 * gamma * field, qubit)

        for qubit in range(num_qubits):
            circuit.rx(2.0 * beta, qubit)

    return circuit


def _z_pauli_label(num_qubits: int, qubits: Sequence[int]) -> str:
    label = ["I"] * num_qubits
    for qubit in qubits:
        label[num_qubits - 1 - int(qubit)] = "Z"
    return "".join(label)


def build_library_qaoa_ansatz(
    pair_weights: np.ndarray,
    fields: np.ndarray,
    depth: int,
) -> Tuple[SparsePauliOp, QAOAAnsatz]:
    num_qubits = pair_weights.shape[0]
    terms: List[Tuple[str, float]] = []

    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            weight = float(pair_weights[i, j])
            if abs(weight) > EPS:
                terms.append((_z_pauli_label(num_qubits, (i, j)), -weight))

    for qubit, field in enumerate(fields):
        field = float(field)
        if abs(field) > EPS:
            terms.append((_z_pauli_label(num_qubits, (qubit,)), -field))

    if not terms:
        terms.append(("I" * num_qubits, 0.0))

    cost_operator = SparsePauliOp.from_list(terms)
    ansatz = QAOAAnsatz(cost_operator=cost_operator, reps=depth, flatten=True)
    return cost_operator, ansatz


def _count_cost_terms(pair_weights: np.ndarray, fields: np.ndarray) -> int:
    pair_terms = int(np.count_nonzero(np.triu(np.abs(pair_weights) > EPS, k=1)))
    field_terms = int(np.count_nonzero(np.abs(fields) > EPS))
    return max(1, pair_terms + field_terms)


def _qaoa_objective(probabilities: np.ndarray, pair_weights: np.ndarray, fields: np.ndarray) -> float:
    spins = basis_spins(pair_weights.shape[0])
    quadratic = 0.5 * np.einsum("bi,ij,bj->b", spins, pair_weights, spins, optimize=True)
    linear = spins @ fields
    cost = -(quadratic + linear)
    return float(probabilities @ cost)


def _standard_qaoa_gate_counts(
    pair_weights: np.ndarray,
    fields: np.ndarray,
    depth: int,
) -> tuple[int, int]:
    pair_terms = int(np.count_nonzero(np.triu(np.abs(pair_weights) > EPS, k=1)))
    field_terms = int(np.count_nonzero(np.abs(fields) > EPS))
    one_qubit_gate_count = pair_weights.shape[0] + depth * (field_terms + pair_weights.shape[0])
    two_qubit_gate_count = depth * pair_terms
    return one_qubit_gate_count, two_qubit_gate_count


def optimize_standard_qaoa(
    pair_weights: np.ndarray,
    fields: np.ndarray,
    depth: int,
    restarts: int,
    maxiter: int,
    seed: int,
    *,
    error_model: str = "ideal",
    error_model_specs_path: str = "Ankaa-3_device_specs.csv",
    error_model_layout_seed: int = 0,
    optimizer_config: StandardQAOAOptimizerConfig | None = None,
) -> Tuple[np.ndarray, Statevector]:
    optimizer_config = StandardQAOAOptimizerConfig() if optimizer_config is None else optimizer_config
    rng = np.random.default_rng(seed)
    best_value = -math.inf
    best_params = None
    best_state = None
    term_count = _count_cost_terms(pair_weights, fields)
    one_qubit_gate_count, two_qubit_gate_count = _standard_qaoa_gate_counts(pair_weights, fields, depth)

    def objective(params: np.ndarray) -> float:
        qaoa = build_standard_qaoa_circuit(pair_weights, fields, params[:depth], params[depth:])
        record_quantum_statevector_evaluation(
            num_qubits=pair_weights.shape[0],
            depth=depth,
            observable_terms=term_count,
            evaluation_kind="objective",
        )
        probabilities, _ = simulate_circuit_probabilities_with_error_model(
            qaoa,
            num_qubits=pair_weights.shape[0],
            one_qubit_gate_count=one_qubit_gate_count,
            two_qubit_gate_count=two_qubit_gate_count,
            measurement_qubits=pair_weights.shape[0],
            error_model=error_model,
            specs_path=error_model_specs_path,
            layout_seed=error_model_layout_seed,
        )
        return -_qaoa_objective(probabilities, pair_weights, fields)

    for restart in range(max(1, restarts)):
        if restart == 0:
            x0 = np.concatenate(
                [
                    np.full(depth, optimizer_config.initial_gamma, dtype=float),
                    np.full(depth, optimizer_config.initial_beta, dtype=float),
                ]
            )
        else:
            x0 = np.concatenate(
                [
                    rng.uniform(
                        optimizer_config.random_gamma_min,
                        optimizer_config.random_gamma_max,
                        size=depth,
                    ),
                    rng.uniform(
                        optimizer_config.random_beta_min,
                        optimizer_config.random_beta_max,
                        size=depth,
                    ),
                ]
            )

        result = minimize(
            objective,
            x0,
            method=optimizer_config.method,
            options={
                "maxiter": maxiter,
                "xatol": optimizer_config.xatol,
                "fatol": optimizer_config.fatol,
                "adaptive": optimizer_config.adaptive,
            },
        )

        params = np.asarray(result.x, dtype=float)
        record_quantum_statevector_evaluation(
            num_qubits=pair_weights.shape[0],
            depth=depth,
            observable_terms=term_count,
            evaluation_kind="objective",
        )
        final_circuit = build_standard_qaoa_circuit(pair_weights, fields, params[:depth], params[depth:])
        probabilities, _ = simulate_circuit_probabilities_with_error_model(
            final_circuit,
            num_qubits=pair_weights.shape[0],
            one_qubit_gate_count=one_qubit_gate_count,
            two_qubit_gate_count=two_qubit_gate_count,
            measurement_qubits=pair_weights.shape[0],
            error_model=error_model,
            specs_path=error_model_specs_path,
            layout_seed=error_model_layout_seed,
        )
        value = _qaoa_objective(probabilities, pair_weights, fields)
        if value > best_value:
            best_value = value
            best_params = params
            best_state = Statevector.from_instruction(final_circuit)

    if best_params is None or best_state is None:
        raise RuntimeError("Standard QAOA baseline failed to optimize parameters.")
    return best_params, best_state


def optimize_library_qaoa(
    pair_weights: np.ndarray,
    fields: np.ndarray,
    depth: int,
    restarts: int,
    maxiter: int,
    seed: int,
    *,
    error_model: str = "ideal",
    error_model_specs_path: str = "Ankaa-3_device_specs.csv",
    error_model_layout_seed: int = 0,
    optimizer_config: LibraryQAOAOptimizerConfig | None = None,
) -> Tuple[np.ndarray, Statevector]:
    if depth < 1:
        raise ValueError("QAOA depth must be at least 1.")

    optimizer_config = LibraryQAOAOptimizerConfig() if optimizer_config is None else optimizer_config
    rng = np.random.default_rng(seed)
    cost_operator, ansatz = build_library_qaoa_ansatz(pair_weights, fields, depth)
    best_value = -math.inf
    best_params = None
    best_state = None
    term_count = max(1, len(cost_operator))
    one_qubit_gate_count, two_qubit_gate_count = _standard_qaoa_gate_counts(pair_weights, fields, depth)

    def objective(params: np.ndarray) -> float:
        record_quantum_statevector_evaluation(
            num_qubits=pair_weights.shape[0],
            depth=depth,
            observable_terms=term_count,
            evaluation_kind="objective",
        )
        probabilities, _ = simulate_circuit_probabilities_with_error_model(
            ansatz.assign_parameters(params),
            num_qubits=pair_weights.shape[0],
            one_qubit_gate_count=one_qubit_gate_count,
            two_qubit_gate_count=two_qubit_gate_count,
            measurement_qubits=pair_weights.shape[0],
            error_model=error_model,
            specs_path=error_model_specs_path,
            layout_seed=error_model_layout_seed,
        )
        return -_qaoa_objective(probabilities, pair_weights, fields)

    for restart in range(max(1, restarts)):
        if restart == 0:
            x0 = np.concatenate(
                [
                    np.full(depth, optimizer_config.initial_cost_angle, dtype=float),
                    np.full(depth, optimizer_config.initial_mixer_angle, dtype=float),
                ]
            )
        else:
            x0 = np.concatenate(
                [
                    rng.uniform(
                        optimizer_config.random_cost_angle_min,
                        optimizer_config.random_cost_angle_max,
                        size=depth,
                    ),
                    rng.uniform(
                        optimizer_config.random_mixer_angle_min,
                        optimizer_config.random_mixer_angle_max,
                        size=depth,
                    ),
                ]
            )

        result = minimize(
            objective,
            x0,
            method=optimizer_config.method,
            options={
                "maxiter": maxiter,
                "xatol": optimizer_config.xatol,
                "fatol": optimizer_config.fatol,
                "adaptive": optimizer_config.adaptive,
            },
        )

        params = np.asarray(result.x, dtype=float)
        record_quantum_statevector_evaluation(
            num_qubits=pair_weights.shape[0],
            depth=depth,
            observable_terms=term_count,
            evaluation_kind="objective",
        )
        final_circuit = ansatz.assign_parameters(params)
        probabilities, _ = simulate_circuit_probabilities_with_error_model(
            final_circuit,
            num_qubits=pair_weights.shape[0],
            one_qubit_gate_count=one_qubit_gate_count,
            two_qubit_gate_count=two_qubit_gate_count,
            measurement_qubits=pair_weights.shape[0],
            error_model=error_model,
            specs_path=error_model_specs_path,
            layout_seed=error_model_layout_seed,
        )
        score = _qaoa_objective(probabilities, pair_weights, fields)
        if score > best_value:
            best_value = score
            best_params = params
            best_state = Statevector.from_instruction(final_circuit)

    if best_params is None or best_state is None:
        raise RuntimeError("Qiskit library QAOA baseline failed to optimize parameters.")
    return best_params, best_state


def most_likely_spin_assignment(num_qubits: int, state: Statevector) -> np.ndarray:
    record_quantum_decode(num_qubits)
    probabilities = np.abs(state.data) ** 2
    index = int(np.argmax(probabilities))
    bits = np.array([(index >> qubit) & 1 for qubit in range(num_qubits)], dtype=int)
    return 1 - 2 * bits


def apply_exact_block_update(
    graph: nx.Graph,
    block: Sequence[int],
    assignment: Dict[int, int],
    seed: int | None = None,
) -> None:
    del seed
    problem = build_local_problem(graph, block, assignment)
    block_spins = exact_block_assignment(
        problem.original_pair_weights,
        problem.original_fields,
        problem.original_pair_weights,
        problem.original_fields,
    )
    for node, spin in zip(problem.nodes, block_spins):
        assignment[node] = int(spin)


def apply_greedy_block_update(
    graph: nx.Graph,
    block: Sequence[int],
    assignment: Dict[int, int],
    seed: int | None = None,
) -> None:
    problem = build_local_problem(graph, block, assignment)
    record_classical_greedy_block(problem.size)
    spins = np.array([assignment[node] for node in problem.nodes], dtype=int)
    order = list(range(problem.size))
    rng = random.Random(0 if seed is None else seed)

    for _ in range(max(1, problem.size)):
        rng.shuffle(order)
        improved = False
        for index in order:
            local_field = float(problem.original_fields[index] + problem.original_pair_weights[index] @ spins)
            delta_energy = -2.0 * spins[index] * local_field
            if delta_energy < -1e-12:
                spins[index] *= -1
                improved = True
        if not improved:
            break

    for node, spin in zip(problem.nodes, spins):
        assignment[node] = int(spin)


def apply_quantum_block_update(
    graph: nx.Graph,
    block: Sequence[int],
    assignment: Dict[int, int],
    *,
    depth: int,
    restarts: int,
    maxiter: int,
    seed: int,
    use_warm_start: bool,
    use_library_qaoa: bool,
    error_model: str = "ideal",
    error_model_specs_path: str = "Ankaa-3_device_specs.csv",
    error_model_layout_seed: int = 0,
    standard_optimizer_config: StandardQAOAOptimizerConfig | None = None,
    library_optimizer_config: LibraryQAOAOptimizerConfig | None = None,
    warm_start_optimizer_config: WarmStartQAOAOptimizerConfig | None = None,
) -> None:
    problem = build_local_problem(graph, block, assignment)
    record_quantum_block_update(problem.size)

    if use_warm_start:
        stats = optimize_warm_start_qaoa(
            problem,
            depth=depth,
            restarts=restarts,
            maxiter=maxiter,
            seed=seed,
            error_model=error_model,
            error_model_specs_path=error_model_specs_path,
            error_model_layout_seed=error_model_layout_seed,
            optimizer_config=warm_start_optimizer_config,
        )
        record_quantum_statevector_evaluation(
            num_qubits=problem.size,
            depth=depth,
            observable_terms=max(1, problem.size),
            evaluation_kind="decode",
        )
        decode_circuit = build_warm_start_qaoa(problem, stats.params[:depth], stats.params[depth:])
        pair_terms = int(np.count_nonzero(np.triu(np.abs(problem.scaled_pair_weights) > EPS, k=1)))
        field_terms = int(np.count_nonzero(np.abs(problem.scaled_fields) > EPS))
        one_qubit_gate_count = problem.size + depth * (field_terms + 3 * problem.size)
        two_qubit_gate_count = depth * pair_terms
    elif use_library_qaoa:
        params, _ = optimize_library_qaoa(
            problem.scaled_pair_weights,
            problem.scaled_fields,
            depth=depth,
            restarts=restarts,
            maxiter=maxiter,
            seed=seed,
            error_model=error_model,
            error_model_specs_path=error_model_specs_path,
            error_model_layout_seed=error_model_layout_seed,
            optimizer_config=library_optimizer_config,
        )
        _, ansatz = build_library_qaoa_ansatz(
            problem.scaled_pair_weights,
            problem.scaled_fields,
            depth,
        )
        decode_circuit = ansatz.assign_parameters(params)
        one_qubit_gate_count, two_qubit_gate_count = _standard_qaoa_gate_counts(
            problem.scaled_pair_weights,
            problem.scaled_fields,
            depth,
        )
    else:
        params, _ = optimize_standard_qaoa(
            problem.scaled_pair_weights,
            problem.scaled_fields,
            depth=depth,
            restarts=restarts,
            maxiter=maxiter,
            seed=seed,
            error_model=error_model,
            error_model_specs_path=error_model_specs_path,
            error_model_layout_seed=error_model_layout_seed,
            optimizer_config=standard_optimizer_config,
        )
        decode_circuit = build_standard_qaoa_circuit(
            problem.scaled_pair_weights,
            problem.scaled_fields,
            params[:depth],
            params[depth:],
        )
        one_qubit_gate_count, two_qubit_gate_count = _standard_qaoa_gate_counts(
            problem.scaled_pair_weights,
            problem.scaled_fields,
            depth,
        )

    probabilities, _ = simulate_circuit_probabilities_with_error_model(
        decode_circuit,
        num_qubits=problem.size,
        one_qubit_gate_count=one_qubit_gate_count,
        two_qubit_gate_count=two_qubit_gate_count,
        measurement_qubits=problem.size,
        error_model=error_model,
        specs_path=error_model_specs_path,
        layout_seed=error_model_layout_seed,
    )
    index = int(np.argmax(probabilities))
    record_quantum_decode(problem.size)
    bits = np.array([(index >> qubit) & 1 for qubit in range(problem.size)], dtype=int)
    block_spins = 1 - 2 * bits
    for node, spin in zip(problem.nodes, block_spins):
        assignment[node] = int(spin)


def partitioned_quantum_baseline(
    graph: nx.Graph,
    graph_name: str,
    method: str,
    depth: int = 1,
    max_block_size: int = 12,
    rounds: int = 1,
    restarts: int = 1,
    maxiter: int = 12,
    seed: int = 0,
    exact_cut_value: float | None = None,
    use_warm_start: bool = False,
    use_library_qaoa: bool = False,
    match_hybrid_schedule: bool = False,
    standard_optimizer_config: StandardQAOAOptimizerConfig | None = None,
    library_optimizer_config: LibraryQAOAOptimizerConfig | None = None,
    warm_start_optimizer_config: WarmStartQAOAOptimizerConfig | None = None,
) -> BaselineResult:
    assignment = random_assignment(graph, seed)
    assignment = greedy_refine(graph, assignment, seed=seed)
    metadata: Dict[str, object] = {
        "depth": depth,
        "max_block_size": max_block_size,
        "rounds": rounds,
        "restarts": restarts,
        "maxiter": maxiter,
    }

    if match_hybrid_schedule:
        schedule = build_partition_schedule(
            graph,
            max_block_size=max_block_size,
            seed=seed,
            strategy="multilevel",
        )
        if schedule.layout is None:
            raise ValueError("The matched hybrid schedule requires a multilevel partition layout.")
        metadata.update(
            {
                "partition_mode": "matched_hybrid_schedule",
                "coarse_nodes": schedule.layout.coarse_graph.number_of_nodes(),
                "boundary_nodes": len(schedule.layout.boundary_nodes),
                "region_blocks": len(schedule.region_blocks),
                "quantum_blocks": len(schedule.boundary_blocks),
            }
        )
    else:
        schedule = build_partition_schedule(
            graph,
            max_block_size=max_block_size,
            seed=seed,
            strategy="recursive",
        )
        metadata["partition_mode"] = "recursive"

    def quantum_updater(
        run_graph: nx.Graph,
        block: Sequence[int],
        current_assignment: Dict[int, int],
        local_seed: int,
    ) -> None:
        apply_quantum_block_update(
            run_graph,
            block,
            current_assignment,
            depth=depth,
            restarts=restarts,
            maxiter=maxiter,
            seed=local_seed,
            use_warm_start=use_warm_start,
            use_library_qaoa=use_library_qaoa,
            standard_optimizer_config=standard_optimizer_config,
            library_optimizer_config=library_optimizer_config,
            warm_start_optimizer_config=warm_start_optimizer_config,
        )

    pipeline_result = run_block_coordinate_descent(
        graph,
        assignment,
        region_blocks=schedule.region_blocks,
        boundary_blocks=schedule.boundary_blocks,
        rounds=rounds,
        seed=seed,
        region_updater=apply_exact_block_update if match_hybrid_schedule else quantum_updater,
        boundary_updater=quantum_updater if match_hybrid_schedule else None,
        refine_assignment=lambda run_graph, current, refine_seed: greedy_refine(run_graph, current, seed=refine_seed),
        objective_value=weighted_cut_value,
        shuffle_region_blocks=not match_hybrid_schedule,
    )
    assignment = pipeline_result.assignment
    metadata["rounds_run"] = pipeline_result.rounds_run

    if use_warm_start:
        implementation = "warm_start_custom"
    elif use_library_qaoa:
        implementation = "qiskit_qaoa_ansatz"
    else:
        implementation = "custom_standard"

    return BaselineResult(
        graph_name=graph_name,
        method=method,
        family="quantum",
        assignment=assignment,
        cut_value=weighted_cut_value(graph, assignment),
        ising_energy=ising_energy(graph, assignment),
        exact_cut_value=exact_cut_value,
        metadata={**metadata, "implementation": implementation},
    )


def compare_methods(
    paths: Sequence[Path] | None = None,
    *,
    root: Path | None = None,
    exact_threshold: int = 21,
    seed: int = 7,
    settings: ComparisonSettings | None = None,
    hybrid_results: Sequence[SolveResult] | None = None,
) -> pd.DataFrame:
    settings = get_comparison_settings() if settings is None else settings

    if paths is None:
        root = Path.cwd() if root is None else Path(root)
        paths = discover_parquet_files(root)
    else:
        paths = [Path(path) for path in paths]

    loaded: List[Tuple[Path, nx.Graph]] = []
    for path in paths:
        resolved = path.resolve()
        loaded.append((resolved, load_weighted_graph(resolved)))

    loaded.sort(key=lambda item: item[1].number_of_edges())
    hybrid_by_name: Dict[str, SolveResult] = {}
    for result in hybrid_results or []:
        graph_name = result.name.split(" :: ")[-1]
        hybrid_by_name[graph_name] = result

    rows: List[Dict[str, object]] = []

    for path, graph in loaded:
        graph_name = path.name
        exact_cut_value = exact_cut_for_graph(graph, threshold=exact_threshold)

        methods: List[BaselineResult] = [
            random_search_baseline(
                graph,
                graph_name=graph_name,
                trials=settings.random_search_trials,
                seed=seed,
                exact_cut_value=exact_cut_value,
            ),
            networkx_maxcut_baseline(
                graph,
                graph_name=graph_name,
                restarts=settings.networkx_restarts,
                seed=seed,
                exact_cut_value=exact_cut_value,
            ),
            partitioned_networkx_maxcut_baseline(
                graph,
                graph_name=graph_name,
                restarts=settings.networkx_restarts,
                max_region_size=settings.networkx_partitioned_max_region_size,
                rounds=settings.networkx_partitioned_rounds,
                seed=seed + 1,
                exact_cut_value=exact_cut_value,
            ),
        ]

        if settings.include_quantum:
            methods.extend(
                [
                    partitioned_quantum_baseline(
                        graph,
                        graph_name=graph_name,
                        method="Standard QAOA (Partitioned)",
                        depth=settings.standard_qaoa_depth,
                        max_block_size=settings.standard_qaoa_max_block_size,
                        rounds=settings.standard_qaoa_rounds,
                        restarts=settings.standard_qaoa_restarts,
                        maxiter=settings.standard_qaoa_maxiter,
                        seed=seed + 2,
                        exact_cut_value=exact_cut_value,
                        use_warm_start=False,
                    ),
                    partitioned_quantum_baseline(
                        graph,
                        graph_name=graph_name,
                        method="Qiskit QAOAAnsatz (Matched Hybrid Schedule)",
                        depth=settings.standard_qaoa_depth,
                        max_block_size=settings.standard_qaoa_max_block_size,
                        rounds=settings.standard_qaoa_rounds,
                        restarts=settings.standard_qaoa_restarts,
                        maxiter=settings.standard_qaoa_maxiter,
                        seed=seed + 3,
                        exact_cut_value=exact_cut_value,
                        use_warm_start=False,
                        use_library_qaoa=True,
                        match_hybrid_schedule=True,
                    ),
                    partitioned_quantum_baseline(
                        graph,
                        graph_name=graph_name,
                        method="Warm-Start QAOA (No Preconditioning)",
                        depth=settings.warm_start_qaoa_depth,
                        max_block_size=settings.warm_start_qaoa_max_block_size,
                        rounds=settings.warm_start_qaoa_rounds,
                        restarts=settings.warm_start_qaoa_restarts,
                        maxiter=settings.warm_start_qaoa_maxiter,
                        seed=seed + 4,
                        exact_cut_value=exact_cut_value,
                        use_warm_start=True,
                    ),
                ]
            )

        if settings.include_hybrid:
            hybrid = hybrid_by_name.get(graph_name)
            if hybrid is None:
                hybrid = solve_graph(
                    graph=graph.copy(),
                    name=graph_name,
                    depth=settings.hybrid_depth,
                    max_block_size=settings.hybrid_max_block_size,
                    rounds=settings.hybrid_rounds,
                    restarts=settings.hybrid_restarts,
                    maxiter=settings.hybrid_maxiter,
                    seed=seed + 5,
                    exact_threshold=exact_threshold,
                )
            methods.append(
                BaselineResult(
                    graph_name=graph_name,
                    method="Standard QAOA Light-Cone Preconditioning + Burer-Monteiro",
                    family="hybrid",
                    assignment=hybrid.assignment,
                    cut_value=hybrid.cut_value,
                    ising_energy=hybrid.ising_energy,
                    exact_cut_value=hybrid.exact_cut_value if hybrid.exact_cut_value is not None else exact_cut_value,
                    metadata={
                        "profile": settings.profile,
                        "depth": settings.hybrid_depth,
                        "max_block_size": settings.hybrid_max_block_size,
                        "rounds": settings.hybrid_rounds,
                        "restarts": settings.hybrid_restarts,
                        "maxiter": settings.hybrid_maxiter,
                    },
                )
            )

        for result in methods:
            row = result.to_dict()
            row["nodes"] = graph.number_of_nodes()
            row["edges"] = graph.number_of_edges()
            row["profile"] = settings.profile
            rows.append(row)

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["graph_name", "cut_value"], ascending=[True, False]).reset_index(drop=True)
    return frame
