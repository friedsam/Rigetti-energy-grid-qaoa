from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from scipy.optimize import minimize

from ..error_modeling.models import simulate_circuit_probabilities_with_error_model
from ..graph_ops import greedy_refine, initial_assignment, ising_energy, weighted_cut_value
from ..partitioning_methods.strategies import build_partition_schedule
from ..pipeline import run_block_coordinate_descent
from ..resource_accounting import (
    record_classical_exact_block,
    record_global_exact,
    record_quantum_block_update,
    record_quantum_statevector_evaluation,
)
from .hyperparameters import WarmStartQAOAOptimizerConfig


EPS = 1e-9


@dataclass
class LocalProblem:
    nodes: Tuple[int, ...]
    pair_weights: np.ndarray
    fields: np.ndarray
    original_pair_weights: np.ndarray
    original_fields: np.ndarray
    warm_probabilities: np.ndarray
    scaled_pair_weights: np.ndarray
    scaled_fields: np.ndarray
    scale: float

    @property
    def size(self) -> int:
        return len(self.nodes)


@dataclass
class QAOAStats:
    params: np.ndarray
    objective: float
    magnetizations: np.ndarray
    correlations: np.ndarray


@dataclass
class SolveResult:
    name: str
    graph: nx.Graph
    assignment: Dict[int, int]
    blocks: List[Tuple[int, ...]]
    rounds_run: int
    cut_value: float
    ising_energy: float
    exact_cut_value: float | None
    coarse_nodes: int = 0
    boundary_nodes: int = 0
    boundary_blocks: int = 0
    strategy: str = "multilevel_region_qaoa"

    def to_dict(self) -> Dict[str, object]:
        plus, minus = render_partition(self)
        return {
            "name": self.name,
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "blocks": len(self.blocks),
            "rounds_run": self.rounds_run,
            "weighted_cut": self.cut_value,
            "ising_energy": self.ising_energy,
            "exact_cut_value": self.exact_cut_value,
            "coarse_nodes": self.coarse_nodes,
            "boundary_nodes": self.boundary_nodes,
            "boundary_blocks": self.boundary_blocks,
            "strategy": self.strategy,
            "partition_plus": plus,
            "partition_minus": minus,
        }


def discover_parquet_files(root: Path) -> List[Path]:
    return sorted(root.glob("*.parquet"), key=lambda path: path.stat().st_size)


def load_weighted_graph(path: Path) -> nx.Graph:
    df = pd.read_parquet(path)
    required = {"node_1", "node_2", "weight"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")

    graph = nx.from_pandas_edgelist(
        df,
        source="node_1",
        target="node_2",
        edge_attr="weight",
        create_using=nx.Graph(),
    )
    relabel = {node: int(node) for node in graph.nodes}
    graph = nx.relabel_nodes(graph, relabel)
    for u, v, data in graph.edges(data=True):
        data["weight"] = float(data["weight"])
    return graph


def normalize_signed(values: np.ndarray) -> np.ndarray:
    scale = float(np.max(np.abs(values))) if values.size else 0.0
    if scale < EPS:
        return np.zeros_like(values, dtype=float)
    return np.clip(values / scale, -1.0, 1.0)


def local_spectral_scores(subgraph: nx.Graph, nodes: Sequence[int]) -> np.ndarray:
    n = len(nodes)
    if n <= 1:
        return np.zeros(n, dtype=float)

    laplacian = nx.laplacian_matrix(subgraph, nodelist=nodes, weight="weight").astype(float)
    dense = laplacian.toarray()
    if np.allclose(dense, 0.0):
        return np.zeros(n, dtype=float)

    eigenvalues, eigenvectors = np.linalg.eigh(dense)
    index = 1 if n > 1 else 0
    vector = np.real(eigenvectors[:, index])
    if np.max(np.abs(vector)) < EPS:
        vector = np.real(eigenvectors[:, -1])
    return normalize_signed(vector)


def build_local_problem(
    graph: nx.Graph,
    block: Sequence[int],
    assignment: Dict[int, int],
) -> LocalProblem:
    nodes = tuple(block)
    n = len(nodes)
    pair_weights = np.zeros((n, n), dtype=float)
    fields = np.zeros(n, dtype=float)
    node_to_index = {node: index for index, node in enumerate(nodes)}

    for i, node in enumerate(nodes):
        for neighbor, data in graph[node].items():
            weight = float(data["weight"])
            j = node_to_index.get(neighbor)
            if j is None:
                fields[i] += weight * assignment[neighbor]
                continue
            if i < j:
                pair_weights[i, j] = weight
                pair_weights[j, i] = weight

    subgraph = graph.subgraph(nodes).copy()
    spectral = local_spectral_scores(subgraph, nodes)
    current = np.array([assignment[node] for node in nodes], dtype=float)
    current = normalize_signed(current)
    field_preference = normalize_signed(-fields)
    if np.dot(spectral, current) < 0.0:
        spectral = -spectral

    blended = normalize_signed(0.60 * spectral + 0.25 * current + 0.15 * field_preference)
    warm_probabilities = np.clip(0.5 * (blended + 1.0), 0.10, 0.90)

    scale = max(
        1.0,
        float(np.max(np.abs(pair_weights))) if pair_weights.size else 0.0,
        float(np.max(np.abs(fields))) if fields.size else 0.0,
    )

    return LocalProblem(
        nodes=nodes,
        pair_weights=pair_weights,
        fields=fields,
        original_pair_weights=pair_weights.copy(),
        original_fields=fields.copy(),
        warm_probabilities=warm_probabilities,
        scaled_pair_weights=pair_weights / scale,
        scaled_fields=fields / scale,
        scale=scale,
    )


@lru_cache(maxsize=None)
def basis_spins(n: int) -> np.ndarray:
    basis_size = 1 << n
    indices = np.arange(basis_size, dtype=np.uint32)[:, None]
    bits = (indices >> np.arange(n, dtype=np.uint32)) & 1
    return 1.0 - 2.0 * bits.astype(float)


def cost_objective_values(pair_weights: np.ndarray, fields: np.ndarray) -> np.ndarray:
    spins = basis_spins(pair_weights.shape[0])
    quadratic = 0.5 * np.einsum("bi,ij,bj->b", spins, pair_weights, spins, optimize=True)
    linear = spins @ fields
    return -(quadratic + linear)


def count_cost_terms(pair_weights: np.ndarray, fields: np.ndarray) -> int:
    pair_terms = int(np.count_nonzero(np.triu(np.abs(pair_weights) > EPS, k=1)))
    field_terms = int(np.count_nonzero(np.abs(fields) > EPS))
    return max(1, pair_terms + field_terms)


def build_warm_start_qaoa(
    problem: LocalProblem,
    gammas: np.ndarray,
    betas: np.ndarray,
) -> QuantumCircuit:
    circuit = QuantumCircuit(problem.size)
    thetas = 2.0 * np.arcsin(np.sqrt(problem.warm_probabilities))

    for qubit, theta in enumerate(thetas):
        circuit.ry(float(theta), qubit)

    for layer in range(len(gammas)):
        gamma = float(gammas[layer])
        beta = float(betas[layer])

        for i in range(problem.size):
            for j in range(i + 1, problem.size):
                weight = problem.scaled_pair_weights[i, j]
                if abs(weight) > EPS:
                    circuit.rzz(-2.0 * gamma * float(weight), i, j)

        for qubit, field in enumerate(problem.scaled_fields):
            if abs(field) > EPS:
                circuit.rz(-2.0 * gamma * float(field), qubit)

        for qubit, theta in enumerate(thetas):
            circuit.ry(float(-theta), qubit)
            circuit.rz(float(-2.0 * beta), qubit)
            circuit.ry(float(theta), qubit)

    return circuit


def qaoa_statistics(
    problem: LocalProblem,
    params: np.ndarray,
    *,
    error_model: str = "ideal",
    error_model_specs_path: str = "Ankaa-3_device_specs.csv",
    error_model_layout_seed: int = 0,
) -> QAOAStats:
    depth = len(params) // 2
    gammas = params[:depth]
    betas = params[depth:]
    circuit = build_warm_start_qaoa(problem, gammas, betas)
    postprocess_terms = count_cost_terms(problem.scaled_pair_weights, problem.scaled_fields) + problem.size + (problem.size * problem.size)
    record_quantum_statevector_evaluation(
        num_qubits=problem.size,
        depth=depth,
        observable_terms=postprocess_terms,
        evaluation_kind="objective",
    )
    pair_terms = int(np.count_nonzero(np.triu(np.abs(problem.scaled_pair_weights) > EPS, k=1)))
    field_terms = int(np.count_nonzero(np.abs(problem.scaled_fields) > EPS))
    one_qubit_gate_count = problem.size + depth * (field_terms + 3 * problem.size)
    two_qubit_gate_count = depth * pair_terms
    probabilities, _ = simulate_circuit_probabilities_with_error_model(
        circuit,
        num_qubits=problem.size,
        one_qubit_gate_count=one_qubit_gate_count,
        two_qubit_gate_count=two_qubit_gate_count,
        measurement_qubits=problem.size,
        error_model=error_model,
        specs_path=error_model_specs_path,
        layout_seed=error_model_layout_seed,
    )
    spins = basis_spins(problem.size)
    objective_values = cost_objective_values(problem.scaled_pair_weights, problem.scaled_fields)
    objective = float(probabilities @ objective_values)
    magnetizations = probabilities @ spins
    correlations = spins.T @ (spins * probabilities[:, None])
    return QAOAStats(
        params=params.copy(),
        objective=objective,
        magnetizations=magnetizations,
        correlations=correlations,
    )


def optimize_warm_start_qaoa(
    problem: LocalProblem,
    depth: int,
    restarts: int,
    maxiter: int,
    seed: int,
    *,
    error_model: str = "ideal",
    error_model_specs_path: str = "Ankaa-3_device_specs.csv",
    error_model_layout_seed: int = 0,
    optimizer_config: WarmStartQAOAOptimizerConfig | None = None,
) -> QAOAStats:
    if depth < 1:
        raise ValueError("QAOA depth must be at least 1.")

    optimizer_config = WarmStartQAOAOptimizerConfig() if optimizer_config is None else optimizer_config
    rng = np.random.default_rng(seed)
    best: QAOAStats | None = None

    def objective(params: np.ndarray) -> float:
        return -qaoa_statistics(
            problem,
            params,
            error_model=error_model,
            error_model_specs_path=error_model_specs_path,
            error_model_layout_seed=error_model_layout_seed,
        ).objective

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

        stats = qaoa_statistics(
            problem,
            np.asarray(result.x, dtype=float),
            error_model=error_model,
            error_model_specs_path=error_model_specs_path,
            error_model_layout_seed=error_model_layout_seed,
        )
        if best is None or stats.objective > best.objective:
            best = stats

    if best is None:
        raise RuntimeError("Warm-start QAOA optimization did not produce a result.")
    return best


def exact_block_assignment(
    pair_weights: np.ndarray,
    fields: np.ndarray,
    tie_pair_weights: np.ndarray,
    tie_fields: np.ndarray,
) -> np.ndarray:
    block_size = pair_weights.shape[0]
    state_count = 1 << block_size
    spins = basis_spins(pair_weights.shape[0])
    energy = 0.5 * np.einsum("bi,ij,bj->b", spins, pair_weights, spins, optimize=True) + spins @ fields
    best_value = float(np.min(energy))
    candidate_indices = np.flatnonzero(np.isclose(energy, best_value, atol=1e-9))
    if len(candidate_indices) == 1:
        record_classical_exact_block(block_size=block_size, states=state_count)
        return spins[int(candidate_indices[0])].copy()

    tie_energy = (
        0.5 * np.einsum("bi,ij,bj->b", spins, tie_pair_weights, spins, optimize=True)
        + spins @ tie_fields
    )
    record_classical_exact_block(
        block_size=block_size,
        states=state_count,
        tiebreak_states=state_count,
    )
    best_index = min(candidate_indices, key=lambda idx: float(tie_energy[int(idx)]))
    return spins[int(best_index)].copy()


def solve_small_graph(
    graph: nx.Graph,
    seed: int,
    exact_threshold: int,
) -> Dict[int, int]:
    if graph.number_of_nodes() == 0:
        return {}
    if graph.number_of_nodes() == 1:
        node = next(iter(graph.nodes()))
        return {int(node): 1}
    if graph.number_of_nodes() <= exact_threshold:
        _, assignment = chunked_exact_cut(graph)
        return assignment

    assignment = initial_assignment(graph, seed)
    return greedy_refine(graph, assignment, seed=seed)


def assign_regions_from_coarse(
    regions: Sequence[Sequence[int]],
    coarse_assignment: Dict[int, int],
) -> Dict[int, int]:
    assignment: Dict[int, int] = {}
    for region_index, region in enumerate(regions):
        sign = int(coarse_assignment.get(region_index, 1))
        for node in region:
            assignment[int(node)] = sign
    return assignment


def solve_node_block(
    graph: nx.Graph,
    nodes: Sequence[int],
    assignment: Dict[int, int],
    depth: int,
    restarts: int,
    maxiter: int,
    seed: int,
    classical_threshold: int,
    prefer_classical: bool = False,
    optimizer_config: WarmStartQAOAOptimizerConfig | None = None,
) -> None:
    if not nodes:
        return

    problem = build_local_problem(graph, tuple(sorted(nodes)), assignment)
    if prefer_classical or problem.size <= classical_threshold:
        block_solution = exact_block_assignment(
            problem.original_pair_weights,
            problem.original_fields,
            problem.original_pair_weights,
            problem.original_fields,
        )
    else:
        record_quantum_block_update(problem.size)
        stats = optimize_warm_start_qaoa(
            problem,
            depth,
            restarts,
            maxiter,
            seed,
            optimizer_config=optimizer_config,
        )
        preconditioned_pairs = -stats.correlations.copy()
        np.fill_diagonal(preconditioned_pairs, 0.0)
        preconditioned_fields = -stats.magnetizations.copy()
        block_solution = exact_block_assignment(
            preconditioned_pairs,
            preconditioned_fields,
            problem.original_pair_weights,
            problem.original_fields,
        )

    for node, spin in zip(problem.nodes, block_solution):
        assignment[int(node)] = int(spin)


def solve_block_with_preconditioning(
    graph: nx.Graph,
    block: Sequence[int],
    assignment: Dict[int, int],
    depth: int,
    restarts: int,
    maxiter: int,
    seed: int,
    optimizer_config: WarmStartQAOAOptimizerConfig | None = None,
) -> None:
    solve_node_block(
        graph=graph,
        nodes=block,
        assignment=assignment,
        depth=depth,
        restarts=restarts,
        maxiter=maxiter,
        seed=seed,
        classical_threshold=0,
        prefer_classical=False,
        optimizer_config=optimizer_config,
    )


def chunked_exact_cut(graph: nx.Graph, chunk_size: int = 1 << 15) -> Tuple[float, Dict[int, int]]:
    nodes = tuple(sorted(graph.nodes()))
    n = len(nodes)
    record_global_exact(1 << n)
    node_to_index = {node: index for index, node in enumerate(nodes)}
    edges = [
        (node_to_index[u], node_to_index[v], float(data["weight"]))
        for u, v, data in graph.edges(data=True)
    ]

    best_cut = -math.inf
    best_assignment = None
    total_states = 1 << n

    for start in range(0, total_states, chunk_size):
        stop = min(total_states, start + chunk_size)
        raw = np.arange(start, stop, dtype=np.uint64)[:, None]
        bits = (raw >> np.arange(n, dtype=np.uint64)) & 1
        cut_values = np.zeros(stop - start, dtype=float)

        for i, j, weight in edges:
            cut_values += weight * (bits[:, i] ^ bits[:, j])

        local_index = int(np.argmax(cut_values))
        local_best = float(cut_values[local_index])
        if local_best > best_cut:
            best_cut = local_best
            best_bits = bits[local_index]
            best_assignment = {
                node: (1 if int(best_bits[idx]) == 0 else -1)
                for idx, node in enumerate(nodes)
            }

    if best_assignment is None:
        raise RuntimeError("Exact search failed to produce a solution.")
    return best_cut, best_assignment


def solve_graph(
    graph: nx.Graph,
    name: str,
    depth: int,
    max_block_size: int,
    rounds: int,
    restarts: int,
    maxiter: int,
    seed: int,
    exact_threshold: int,
    standard_qaoa_optimizer: "StandardQAOAOptimizerConfig | None" = None,
    library_qaoa_optimizer: "LibraryQAOAOptimizerConfig | None" = None,
    warm_start_qaoa_optimizer: WarmStartQAOAOptimizerConfig | None = None,
    preconditioner_min_abs_weight: float = 1e-6,
) -> SolveResult:
    from .modular import ModularSolveConfig
    from .preconditioning import solve_graph_quantum_preconditioned
    from .hyperparameters import LibraryQAOAOptimizerConfig, StandardQAOAOptimizerConfig

    config = ModularSolveConfig(
        partition_strategy="multilevel",
        region_optimizer="hybrid_preconditioned",
        boundary_optimizer="hybrid_preconditioned",
        depth=depth,
        max_block_size=max_block_size,
        rounds=rounds,
        restarts=restarts,
        maxiter=maxiter,
        seed=seed,
        exact_threshold=exact_threshold,
        allow_exact_postcheck=True,
        standard_qaoa_optimizer=(
            StandardQAOAOptimizerConfig() if standard_qaoa_optimizer is None else standard_qaoa_optimizer
        ),
        library_qaoa_optimizer=(
            LibraryQAOAOptimizerConfig() if library_qaoa_optimizer is None else library_qaoa_optimizer
        ),
        warm_start_qaoa_optimizer=(
            WarmStartQAOAOptimizerConfig() if warm_start_qaoa_optimizer is None else warm_start_qaoa_optimizer
        ),
        preconditioner_min_abs_weight=preconditioner_min_abs_weight,
    )
    return solve_graph_quantum_preconditioned(graph, name=name, config=config)


def render_partition(result: SolveResult) -> Tuple[List[int], List[int]]:
    positive = sorted(node for node, value in result.assignment.items() if value > 0)
    negative = sorted(node for node, value in result.assignment.items() if value < 0)
    return positive, negative


def print_summary(result: SolveResult) -> None:
    positive, negative = render_partition(result)
    print(f"\n[{result.name}]")
    print(
        f"nodes={result.graph.number_of_nodes()} "
        f"edges={result.graph.number_of_edges()} "
        f"regions={len(result.blocks)} "
        f"coarse_nodes={result.coarse_nodes} "
        f"boundary_nodes={result.boundary_nodes} "
        f"boundary_blocks={result.boundary_blocks} "
        f"rounds={result.rounds_run}"
    )
    print(f"weighted_cut={result.cut_value:.6f}")
    print(f"ising_energy={result.ising_energy:.6f}")
    if result.exact_cut_value is not None:
        gap = result.exact_cut_value - result.cut_value
        print(f"exact_cut={result.exact_cut_value:.6f}")
        print(f"exact_gap={gap:.6f}")
    print(f"partition_plus={positive}")
    print(f"partition_minus={negative}")


def solve_challenge(
    paths: Sequence[Path] | None = None,
    *,
    root: Path | None = None,
    depth: int = 1,
    max_block_size: int = 16,
    rounds: int = 3,
    restarts: int = 3,
    maxiter: int = 35,
    seed: int = 7,
    exact_threshold: int = 21,
    standard_qaoa_optimizer: "StandardQAOAOptimizerConfig | None" = None,
    library_qaoa_optimizer: "LibraryQAOAOptimizerConfig | None" = None,
    warm_start_qaoa_optimizer: WarmStartQAOAOptimizerConfig | None = None,
    preconditioner_min_abs_weight: float = 1e-6,
) -> List[SolveResult]:
    if paths is None:
        root = Path.cwd() if root is None else Path(root)
        paths = discover_parquet_files(root)
    else:
        paths = [Path(path) for path in paths]

    if not paths:
        raise FileNotFoundError("No parquet files were found.")

    loaded: List[Tuple[Path, nx.Graph]] = []
    for path in paths:
        resolved = path.resolve()
        loaded.append((resolved, load_weighted_graph(resolved)))

    loaded.sort(key=lambda item: item[1].number_of_edges())
    labels = ["Problem A", "Problem B"]
    results: List[SolveResult] = []

    for index, (path, graph) in enumerate(loaded):
        if len(loaded) == 1:
            name = path.name
        else:
            label = labels[index] if index < len(labels) else f"Problem {index + 1}"
            name = f"{label} :: {path.name}"

        result = solve_graph(
            graph=graph,
            name=name,
            depth=depth,
            max_block_size=max_block_size,
            rounds=rounds,
            restarts=restarts,
            maxiter=maxiter,
            seed=seed + index * 1000,
            exact_threshold=exact_threshold,
            standard_qaoa_optimizer=standard_qaoa_optimizer,
            library_qaoa_optimizer=library_qaoa_optimizer,
            warm_start_qaoa_optimizer=warm_start_qaoa_optimizer,
            preconditioner_min_abs_weight=preconditioner_min_abs_weight,
        )
        results.append(result)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Multilevel quantum-preconditioned solver for the weighted MaxCut challenge. "
            "It builds a standard-QAOA light-cone preconditioned graph and then solves "
            "that transformed problem with classical local search."
        )
    )
    parser.add_argument(
        "--input",
        nargs="*",
        default=None,
        help="Parquet file(s) to solve. If omitted, all parquet files in the working directory are used.",
    )
    parser.add_argument("--depth", type=int, default=1, help="Standard QAOA depth used to build the preconditioner.")
    parser.add_argument(
        "--max-block-size",
        type=int,
        default=16,
        help="Maximum region/local block size used by the multilevel solver.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Maximum number of block-coordinate refinement rounds.",
    )
    parser.add_argument(
        "--restarts",
        type=int,
        default=3,
        help="Random restarts for the classical optimization of QAOA angles per block.",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=35,
        help="Maximum optimizer iterations per QAOA restart.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--exact-threshold",
        type=int,
        default=21,
        help="Run an exact brute-force verification when the graph has at most this many nodes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = [Path(item) for item in args.input] if args.input else None
    for result in solve_challenge(
        paths=paths,
        root=Path.cwd(),
        depth=args.depth,
        max_block_size=args.max_block_size,
        rounds=args.rounds,
        restarts=args.restarts,
        maxiter=args.maxiter,
        seed=args.seed,
        exact_threshold=args.exact_threshold,
    ):
        print_summary(result)


if __name__ == "__main__":
    main()
