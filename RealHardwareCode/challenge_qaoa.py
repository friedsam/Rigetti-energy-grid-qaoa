"""Run Max-Cut region solves on Rigetti-compatible backends and classical fallbacks.

This module contains the hardware-facing workflow used by the challenge code:
it loads the problem graph, builds small induced subproblems, sweeps QAOA angle
landscapes with pyQuil, and falls back to classical or QAOA-preconditioned
solvers when regions exceed the direct hardware budget.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import networkx as nx
import numpy as np
import pandas as pd
from pyquil import Program, get_qc
from pyquil.gates import H, MEASURE
from pyquil.paulis import exponential_map, sX, sZ

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:
    _tqdm = None


DEFAULT_PROBLEM_B_PARQUET = "019c9083-9e69-72f0-b313-85026d9a88aa.parquet"
DEFAULT_QAOA_NODE_LIMIT = 10
EPS = 1e-9


class _NullProgressBar:
    def update(self, _: int = 1) -> None:
        return None

    def close(self) -> None:
        return None


def _progress_iterable(
    iterable: Iterable[Any],
    *,
    enabled: bool,
    total: int | None = None,
    desc: str | None = None,
    leave: bool = False,
) -> Iterable[Any]:
    if enabled and _tqdm is not None:
        return _tqdm(iterable, total=total, desc=desc, leave=leave)
    return iterable


def _progress_bar(
    *,
    enabled: bool,
    total: int | None = None,
    desc: str | None = None,
    leave: bool = False,
) -> Any:
    if enabled and _tqdm is not None:
        return _tqdm(total=total, desc=desc, leave=leave)
    return _NullProgressBar()


def _progress_message(
    message: str,
    *,
    enabled: bool,
) -> None:
    if enabled:
        print(message, flush=True)


@dataclass(frozen=True)
class RegionProblem:
    """A selected induced subgraph plus its qubit-reindexed QAOA representation."""
    original_graph: nx.Graph
    qaoa_graph: nx.Graph
    node_order: tuple[int, ...]


@dataclass(frozen=True)
class PreconditionedGraphBuild:
    """Metadata for a surrogate graph assembled from local QAOA correlation estimates."""
    graph: nx.Graph
    blocks: tuple[tuple[int, ...], ...]
    candidate_edges: int
    correlated_edges: int
    light_cone_evaluations: int
    max_light_cone_size: int


@dataclass(frozen=True)
class SolveResult:
    """Serializable output for one region solve, independent of the chosen method."""
    method: str
    qc_name: str
    selected_nodes: tuple[int, ...]
    assignment: dict[int, int]
    cut_value: float
    left_partition: tuple[int, ...]
    right_partition: tuple[int, ...]
    sample_cut_value: float | None = None
    mean_objective: float | None = None
    beta: float | tuple[float, ...] | None = None
    gamma: float | tuple[float, ...] | None = None
    landscape: np.ndarray | None = None
    beta_values: np.ndarray | None = None
    gamma_values: np.ndarray | None = None
    preconditioned_graph: nx.Graph | None = None
    blocks: tuple[tuple[int, ...], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly payload for reporting or CSV/Notebook export."""
        payload: dict[str, Any] = {
            "method": self.method,
            "qc_name": self.qc_name,
            "selected_nodes": list(self.selected_nodes),
            "cut_value": self.cut_value,
            "left_partition": list(self.left_partition),
            "right_partition": list(self.right_partition),
        }
        if self.sample_cut_value is not None:
            payload["sample_cut_value"] = self.sample_cut_value
        if self.mean_objective is not None:
            payload["mean_objective"] = self.mean_objective
        if self.beta is not None:
            payload["beta"] = _serialize_angle_value(self.beta)
        if self.gamma is not None:
            payload["gamma"] = _serialize_angle_value(self.gamma)
        if self.blocks:
            payload["blocks"] = [list(block) for block in self.blocks]
        if self.preconditioned_graph is not None:
            payload["preconditioned_edge_count"] = int(self.preconditioned_graph.number_of_edges())
        return payload


def _serialize_angle_value(
    value: float | tuple[float, ...] | np.ndarray,
) -> float | list[float]:
    if isinstance(value, np.ndarray):
        flattened = tuple(float(entry) for entry in value.reshape(-1))
        return float(flattened[0]) if len(flattened) == 1 else list(flattened)
    if isinstance(value, tuple):
        return float(value[0]) if len(value) == 1 else [float(entry) for entry in value]
    return float(value)


def resolve_problem_graph_path(parquet_path: str | Path | None = None) -> Path:
    """Resolve the input parquet path, falling back to the default challenge dataset."""
    if parquet_path is not None:
        candidate = Path(parquet_path)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Parquet file not found: {candidate}")

    try:
        from problem_b_mites_regions import resolve_problem_b_parquet

        return resolve_problem_b_parquet()
    except Exception:
        pass

    default_path = Path(DEFAULT_PROBLEM_B_PARQUET)
    if default_path.exists():
        return default_path

    candidates = sorted(Path(".").glob("*.parquet"))
    if not candidates:
        raise FileNotFoundError("No parquet files were found in the current directory.")
    return candidates[0]


def load_problem_graph(parquet_path: str | Path | None = None) -> tuple[Path, nx.Graph]:
    """Load the challenge graph from parquet as a weighted undirected NetworkX graph."""
    path = resolve_problem_graph_path(parquet_path)
    frame = pd.read_parquet(path)

    graph = nx.Graph()
    for row in frame.itertuples(index=False):
        graph.add_edge(int(row.node_1), int(row.node_2), weight=float(row.weight))

    return path, graph


def normalize_selected_nodes(selected_nodes: Sequence[int] | str) -> tuple[int, ...]:
    """Normalize user-supplied node identifiers into a sorted, unique tuple."""
    if isinstance(selected_nodes, str):
        parts = [part for part in re.split(r"[\s,]+", selected_nodes.strip()) if part]
        values = [int(part) for part in parts]
    else:
        values = [int(node) for node in selected_nodes]

    normalized = tuple(sorted(set(values)))
    if not normalized:
        raise ValueError("selected_nodes must contain at least one node.")
    return normalized


def build_region_problem(graph: nx.Graph, selected_nodes: Sequence[int] | str) -> RegionProblem:
    """Extract a region and remap it onto contiguous qubit indices for QAOA execution."""
    node_order = normalize_selected_nodes(selected_nodes)
    missing = [node for node in node_order if node not in graph]
    if missing:
        raise ValueError(f"Nodes not present in graph: {missing}")

    subgraph = graph.subgraph(node_order).copy()
    node_to_qubit = {node: index for index, node in enumerate(node_order)}

    qaoa_graph = nx.Graph()
    qaoa_graph.add_nodes_from(range(len(node_order)))
    for u, v, data in subgraph.edges(data=True):
        qaoa_graph.add_edge(
            node_to_qubit[int(u)],
            node_to_qubit[int(v)],
            weight=float(data.get("weight", 1.0)),
        )

    return RegionProblem(
        original_graph=subgraph,
        qaoa_graph=qaoa_graph,
        node_order=node_order,
    )


def weighted_cut_value(graph: nx.Graph, assignment: dict[int, int]) -> float:
    total = 0.0
    for u, v, data in graph.edges(data=True):
        zi = int(assignment[int(u)])
        zj = int(assignment[int(v)])
        total += 0.5 * float(data.get("weight", 1.0)) * (1.0 - zi * zj)
    return total


def bitstring_cut_value(bitstring: Sequence[int], graph: nx.Graph) -> float:
    ordered_nodes = tuple(sorted(int(node) for node in graph.nodes()))
    assignment = {
        node: 1 if int(bitstring[index]) == 0 else -1
        for index, node in enumerate(ordered_nodes)
    }
    return weighted_cut_value(graph, assignment)


def assignment_from_bitstring(node_order: Sequence[int], bitstring: Sequence[int]) -> dict[int, int]:
    return {
        int(node): 1 if int(bitstring[index]) == 0 else -1
        for index, node in enumerate(node_order)
    }


def build_weighted_maxcut_qaoa_program(
    graph: nx.Graph,
    *,
    layers: int = 1,
    betas: Sequence[float] | None = None,
    gammas: Sequence[float] | None = None,
) -> Program:
    """Build a pyQuil program for weighted Max-Cut QAOA on the provided graph."""
    if graph.number_of_nodes() == 0:
        raise ValueError("Cannot build a QAOA program for an empty graph.")
    layer_count = int(layers)
    if layer_count < 1:
        raise ValueError("layers must be at least 1.")

    ordered_nodes = tuple(sorted(int(node) for node in graph.nodes()))
    program = Program()

    use_concrete_angles = betas is not None or gammas is not None
    if use_concrete_angles:
        if betas is None or gammas is None:
            raise ValueError("betas and gammas must both be provided together.")
        if len(betas) != layer_count or len(gammas) != layer_count:
            raise ValueError("betas and gammas must match the number of layers.")
        beta_registers = tuple(float(value) for value in betas)
        gamma_registers = tuple(float(value) for value in gammas)
    else:
        # Compile a single parametric binary and bind all angles at run time.
        beta_memory = program.declare("beta", "REAL", layer_count)
        gamma_memory = program.declare("gamma", "REAL", layer_count)
        beta_registers = tuple(beta_memory[index] for index in range(layer_count))
        gamma_registers = tuple(gamma_memory[index] for index in range(layer_count))

    ro = program.declare("ro", "BIT", len(ordered_nodes))

    for qubit in ordered_nodes:
        program.inst(H(qubit))

    edge_list = sorted(
        graph.edges(data=True),
        key=lambda edge: (int(edge[0]), int(edge[1])),
    )
    for layer_index in range(layer_count):
        for u, v, data in edge_list:
            weight = float(data.get("weight", 1.0))
            if abs(weight) <= EPS:
                continue
            term = (-0.5 * weight) * sZ(int(u)) * sZ(int(v))
            program.inst(exponential_map(term)(gamma_registers[layer_index]))

        for qubit in ordered_nodes:
            program.inst(exponential_map(sX(int(qubit)))(beta_registers[layer_index]))

    for index, qubit in enumerate(ordered_nodes):
        program.inst(MEASURE(qubit, ro[index]))

    return program


def _extract_readout_matrix(result: Any, register_name: str = "ro") -> np.ndarray:
    if isinstance(result, np.ndarray):
        return np.asarray(result, dtype=int)

    readout_data = getattr(result, "readout_data", None)
    if isinstance(readout_data, dict) and register_name in readout_data:
        return np.asarray(readout_data[register_name], dtype=int)

    data_attr = getattr(result, "data", None)
    result_data = getattr(data_attr, "result_data", None)
    if result_data is not None and hasattr(result_data, "to_register_map"):
        register_map = result_data.to_register_map()
        payload = register_map.get(register_name)
        if hasattr(payload, "to_ndarray"):
            return np.asarray(payload.to_ndarray(), dtype=int)
        if payload is not None:
            return np.asarray(payload, dtype=int)

    if hasattr(result, "get_register_map"):
        register_map = result.get_register_map()
        payload = register_map.get(register_name)
        if payload is not None:
            return np.asarray(payload, dtype=int)

    raise TypeError("Unable to extract readout data from pyQuil run result.")


def _resolve_qc_name(
    num_qubits: int,
    qc_name: str | None = None,
    quantum_processor_id: str | None = None,
) -> str:
    if qc_name:
        return qc_name
    if quantum_processor_id:
        return (
            quantum_processor_id
            if quantum_processor_id.endswith("-qvm")
            else f"{quantum_processor_id}-qvm"
        )
    return f"{max(1, int(num_qubits))}q-qvm"


def _build_qaoa_memory_map(
    betas: Sequence[float],
    gammas: Sequence[float],
) -> dict[str, list[float]]:
    if len(gammas) != len(betas):
        raise ValueError("betas and gammas must have the same number of layers.")
    return {
        "beta": [float(value) for value in betas],
        "gamma": [float(value) for value in gammas],
    }


def _evaluate_qaoa_point(
    qc: Any,
    executable: Any,
    graph: nx.Graph,
    *,
    betas: Sequence[float],
    gammas: Sequence[float],
) -> dict[str, Any]:
    run_result = qc.run(
        executable,
        memory_map=_build_qaoa_memory_map(betas, gammas),
    )
    bitstrings = _extract_readout_matrix(run_result)
    if bitstrings.ndim == 1:
        bitstrings = np.atleast_2d(bitstrings)

    cut_values = np.asarray(
        [bitstring_cut_value(bitstring, graph) for bitstring in bitstrings],
        dtype=float,
    )
    best_index = int(np.argmax(cut_values))
    return {
        "bitstrings": bitstrings,
        "cut_values": cut_values,
        "mean_objective": float(cut_values.mean()),
        "best_cut_value": float(cut_values[best_index]),
        "best_bitstring": tuple(int(bit) for bit in bitstrings[best_index]),
    }


def _is_better_qaoa_point(
    mean_objective: float,
    best_cut_value: float,
    current_mean: float,
    current_cut: float,
) -> bool:
    if mean_objective > current_mean + EPS:
        return True
    if abs(mean_objective - current_mean) <= EPS and best_cut_value > current_cut + EPS:
        return True
    return False


def run_qaoa_landscape(
    graph: nx.Graph,
    *,
    width: int = 21,
    shots: int = 250,
    layers: int = 1,
    seed: int = 42,
    qc_name: str | None = None,
    quantum_processor_id: str | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Sweep a small QAOA angle grid and return the best sampled configuration."""
    if width < 2:
        raise ValueError("width must be at least 2.")
    if shots < 1:
        raise ValueError("shots must be at least 1.")
    if int(layers) < 1:
        raise ValueError("layers must be at least 1.")

    resolved_qc_name = _resolve_qc_name(
        graph.number_of_nodes(),
        qc_name,
        quantum_processor_id,
    )
    qc = get_qc(resolved_qc_name)

    layer_count = int(layers)
    program = build_weighted_maxcut_qaoa_program(
        graph,
        layers=layer_count,
    )
    program.wrap_in_numshots_loop(int(shots))
    executable = qc.compile(program)

    angle_values = np.linspace(0.0, np.pi, int(width))
    landscape = (
        np.zeros((len(angle_values), len(angle_values)), dtype=float)
        if layer_count == 1
        else None
    )
    progress = _progress_bar(
        enabled=show_progress,
        total=(len(angle_values) * len(angle_values)) if layer_count == 1 else None,
        desc="QAOA sweep" if layer_count == 1 else f"QAOA p={layer_count}",
        leave=False,
    )

    best_mean_objective = float("-inf")
    best_sample_cut_value = float("-inf")
    best_cut_value = float("-inf")
    best_bitstring: tuple[int, ...] | None = None
    best_samples: np.ndarray | None = None
    best_betas = tuple(0.0 for _ in range(layer_count))
    best_gammas = tuple(0.0 for _ in range(layer_count))
    try:
        def evaluate_point(
            betas: Sequence[float],
            gammas: Sequence[float],
        ) -> dict[str, Any]:
            point = _evaluate_qaoa_point(
                qc,
                executable,
                graph,
                betas=betas,
                gammas=gammas,
            )
            progress.update(1)
            return point

        if layer_count == 1:
            for gamma_index, gamma in enumerate(angle_values):
                for beta_index, beta in enumerate(angle_values):
                    point = evaluate_point(
                        betas=(float(beta),),
                        gammas=(float(gamma),),
                    )
                    landscape[gamma_index, beta_index] = point["mean_objective"]

                    if _is_better_qaoa_point(
                        point["mean_objective"],
                        point["best_cut_value"],
                        best_mean_objective,
                        best_sample_cut_value,
                    ):
                        best_mean_objective = point["mean_objective"]
                        best_sample_cut_value = point["best_cut_value"]
                        best_betas = (float(beta),)
                        best_gammas = (float(gamma),)
                        best_samples = point["bitstrings"].copy()

                    if point["best_cut_value"] > best_cut_value + EPS:
                        best_cut_value = point["best_cut_value"]
                        best_bitstring = point["best_bitstring"]
        else:
            rng = np.random.default_rng(seed)
            restart_count = max(2, min(8, 2 * layer_count + 1))
            sweep_count = max(2, min(6, 2 * layer_count))
            seeded_schedules: list[tuple[np.ndarray, np.ndarray]] = [
                (
                    np.linspace(np.pi / 4.0, np.pi / 8.0, layer_count, dtype=float),
                    np.linspace(np.pi / 8.0, np.pi / 4.0, layer_count, dtype=float),
                ),
                (
                    np.full(layer_count, np.pi / 4.0, dtype=float),
                    np.full(layer_count, np.pi / 4.0, dtype=float),
                ),
            ]

            for restart_index in range(restart_count):
                if restart_index < len(seeded_schedules):
                    current_betas = seeded_schedules[restart_index][0].copy()
                    current_gammas = seeded_schedules[restart_index][1].copy()
                else:
                    current_betas = rng.choice(angle_values, size=layer_count, replace=True).astype(float)
                    current_gammas = rng.choice(angle_values, size=layer_count, replace=True).astype(float)

                current_point = evaluate_point(
                    betas=current_betas,
                    gammas=current_gammas,
                )

                if _is_better_qaoa_point(
                    current_point["mean_objective"],
                    current_point["best_cut_value"],
                    best_mean_objective,
                    best_sample_cut_value,
                ):
                    best_mean_objective = current_point["mean_objective"]
                    best_sample_cut_value = current_point["best_cut_value"]
                    best_betas = tuple(float(value) for value in current_betas)
                    best_gammas = tuple(float(value) for value in current_gammas)
                    best_samples = current_point["bitstrings"].copy()

                if current_point["best_cut_value"] > best_cut_value + EPS:
                    best_cut_value = current_point["best_cut_value"]
                    best_bitstring = current_point["best_bitstring"]

                for _ in range(sweep_count):
                    improved = False

                    for layer_index in range(layer_count):
                        local_best_point = current_point
                        local_best_value = float(current_gammas[layer_index])
                        for gamma in angle_values:
                            trial_gammas = current_gammas.copy()
                            trial_gammas[layer_index] = float(gamma)
                            point = evaluate_point(
                                betas=current_betas,
                                gammas=trial_gammas,
                            )
                            if _is_better_qaoa_point(
                                point["mean_objective"],
                                point["best_cut_value"],
                                local_best_point["mean_objective"],
                                local_best_point["best_cut_value"],
                            ):
                                local_best_point = point
                                local_best_value = float(gamma)

                            if _is_better_qaoa_point(
                                point["mean_objective"],
                                point["best_cut_value"],
                                best_mean_objective,
                                best_sample_cut_value,
                            ):
                                best_mean_objective = point["mean_objective"]
                                best_sample_cut_value = point["best_cut_value"]
                                best_betas = tuple(float(value) for value in current_betas)
                                best_gammas = tuple(float(value) for value in trial_gammas)
                                best_samples = point["bitstrings"].copy()

                            if point["best_cut_value"] > best_cut_value + EPS:
                                best_cut_value = point["best_cut_value"]
                                best_bitstring = point["best_bitstring"]

                        if local_best_value != float(current_gammas[layer_index]):
                            current_gammas[layer_index] = local_best_value
                            current_point = local_best_point
                            improved = True

                    for layer_index in range(layer_count):
                        local_best_point = current_point
                        local_best_value = float(current_betas[layer_index])
                        for beta in angle_values:
                            trial_betas = current_betas.copy()
                            trial_betas[layer_index] = float(beta)
                            point = evaluate_point(
                                betas=trial_betas,
                                gammas=current_gammas,
                            )
                            if _is_better_qaoa_point(
                                point["mean_objective"],
                                point["best_cut_value"],
                                local_best_point["mean_objective"],
                                local_best_point["best_cut_value"],
                            ):
                                local_best_point = point
                                local_best_value = float(beta)

                            if _is_better_qaoa_point(
                                point["mean_objective"],
                                point["best_cut_value"],
                                best_mean_objective,
                                best_sample_cut_value,
                            ):
                                best_mean_objective = point["mean_objective"]
                                best_sample_cut_value = point["best_cut_value"]
                                best_betas = tuple(float(value) for value in trial_betas)
                                best_gammas = tuple(float(value) for value in current_gammas)
                                best_samples = point["bitstrings"].copy()

                            if point["best_cut_value"] > best_cut_value + EPS:
                                best_cut_value = point["best_cut_value"]
                                best_bitstring = point["best_bitstring"]

                        if local_best_value != float(current_betas[layer_index]):
                            current_betas[layer_index] = local_best_value
                            current_point = local_best_point
                            improved = True

                    if not improved:
                        break
    finally:
        progress.close()

    if best_bitstring is None:
        raise RuntimeError("QAOA sweep did not produce any measurement outcomes.")
    if best_samples is None:
        best_samples = np.asarray([best_bitstring], dtype=int)

    beta_payload: float | tuple[float, ...] = (
        float(best_betas[0]) if layer_count == 1 else tuple(float(value) for value in best_betas)
    )
    gamma_payload: float | tuple[float, ...] = (
        float(best_gammas[0]) if layer_count == 1 else tuple(float(value) for value in best_gammas)
    )

    return {
        "qc_name": resolved_qc_name,
        "landscape": landscape,
        "angle_values": angle_values,
        "best_beta": beta_payload,
        "best_gamma": gamma_payload,
        "best_mean_objective": best_mean_objective,
        "best_cut_value": best_cut_value,
        "best_bitstring": best_bitstring,
        "best_samples": best_samples,
    }


def estimate_correlations(bitstrings: np.ndarray) -> np.ndarray:
    samples = np.asarray(bitstrings, dtype=int)
    if samples.ndim == 1:
        samples = np.atleast_2d(samples)
    spins = 1 - 2 * samples
    return (spins.T @ spins) / float(len(spins))


def _random_assignment(graph: nx.Graph, rng: np.random.Generator) -> dict[int, int]:
    return {
        int(node): 1 if int(rng.integers(0, 2)) == 0 else -1
        for node in graph.nodes()
    }


def greedy_local_search(
    graph: nx.Graph,
    assignment: dict[int, int],
    *,
    max_passes: int = 10_000,
) -> dict[int, int]:
    working = {int(node): int(spin) for node, spin in assignment.items()}

    for _ in range(max_passes):
        best_node: int | None = None
        best_delta = 0.0

        for node in graph.nodes():
            node_id = int(node)
            delta = 0.0
            for neighbor, data in graph[node_id].items():
                delta += (
                    float(data.get("weight", 1.0))
                    * working[node_id]
                    * working[int(neighbor)]
                )
            if delta > best_delta + EPS:
                best_delta = delta
                best_node = node_id

        if best_node is None:
            break

        working[best_node] *= -1

    return working


def multi_start_greedy_maxcut(
    graph: nx.Graph,
    *,
    seed: int = 42,
    restarts: int = 24,
    initial_assignment: dict[int, int] | None = None,
    show_progress: bool = False,
) -> dict[int, int]:
    if graph.number_of_nodes() == 0:
        return {}

    rng = np.random.default_rng(seed)
    best_assignment: dict[int, int] | None = None
    best_value = float("-inf")

    if initial_assignment is not None:
        candidate = greedy_local_search(graph, initial_assignment)
        candidate_value = weighted_cut_value(graph, candidate)
        best_assignment = candidate
        best_value = candidate_value

    for _ in _progress_iterable(
        range(max(1, int(restarts))),
        enabled=show_progress,
        total=max(1, int(restarts)),
        desc="Greedy restarts",
        leave=False,
    ):
        candidate = greedy_local_search(graph, _random_assignment(graph, rng))
        candidate_value = weighted_cut_value(graph, candidate)
        if candidate_value > best_value + EPS:
            best_assignment = candidate
            best_value = candidate_value

    if best_assignment is None:
        raise RuntimeError("Unable to generate a Max-Cut assignment.")
    return best_assignment


def _random_unit_vectors(
    count: int,
    dimension: int,
    rng: np.random.Generator,
) -> np.ndarray:
    vectors = rng.normal(size=(int(count), int(dimension)))
    norms = np.linalg.norm(vectors, axis=1)
    zero_mask = norms <= EPS
    while np.any(zero_mask):
        vectors[zero_mask] = rng.normal(size=(int(np.count_nonzero(zero_mask)), int(dimension)))
        norms = np.linalg.norm(vectors, axis=1)
        zero_mask = norms <= EPS
    return vectors / norms[:, None]


def _normalize_external_command(
    command: Sequence[str] | str | None,
) -> tuple[str, ...]:
    if command is None:
        return ()
    if isinstance(command, str):
        stripped = command.strip()
        return (stripped,) if stripped else ()
    normalized = tuple(str(part).strip() for part in command if str(part).strip())
    return normalized


def _default_mqlib_command() -> tuple[str, ...]:
    for env_name in ("MQLIB_EXECUTABLE", "MQLIB_BM_EXECUTABLE"):
        env_value = os.getenv(env_name)
        if env_value and env_value.strip():
            return (env_value.strip(),)

    for candidate in ("MQLib", "MQLib.exe", "mqlib", "mqlib.exe"):
        resolved = shutil.which(candidate)
        if resolved:
            return (resolved,)
    return ()


def _resolve_mqlib_command(
    command: Sequence[str] | str | None,
) -> tuple[str, ...]:
    explicit = _normalize_external_command(command)
    if explicit:
        return explicit
    discovered = _default_mqlib_command()
    if discovered:
        return discovered
    raise FileNotFoundError(
        "No external MQLib executable was found. Set MQLIB_EXECUTABLE or "
        "pass bm_external_command explicitly."
    )


def _write_mqlib_maxcut_instance(
    graph: nx.Graph,
    path: Path,
) -> tuple[int, ...]:
    ordered_nodes = tuple(sorted(int(node) for node in graph.nodes()))
    node_to_index = {node: index + 1 for index, node in enumerate(ordered_nodes)}
    edges = sorted(
        (
            node_to_index[int(u)],
            node_to_index[int(v)],
            float(data.get("weight", 1.0)),
        )
        for u, v, data in graph.edges(data=True)
    )

    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write(f"{len(ordered_nodes)} {len(edges)}\n")
        for left, right, weight in edges:
            handle.write(f"{left} {right} {weight:.17g}\n")

    return ordered_nodes


def _parse_mqlib_solution(
    stdout: str,
    ordered_nodes: Sequence[int],
) -> dict[int, int]:
    node_count = len(tuple(ordered_nodes))
    marker = "Solution:"
    payload = stdout
    marker_index = stdout.find(marker)
    if marker_index >= 0:
        payload = stdout[marker_index + len(marker):]
    else:
        lines = [line.strip() for line in stdout.splitlines() if line.strip()]
        payload = lines[-1] if lines else ""

    raw_values = [int(token) for token in re.findall(r"[-+]?\d+", payload)]
    if len(raw_values) < node_count:
        raise RuntimeError(
            "External MQLib output did not include enough assignment values to "
            f"cover {node_count} nodes."
        )

    values = raw_values[:node_count]
    unique_values = set(values)
    if unique_values.issubset({0, 1}):
        spins = [1 if value == 1 else -1 for value in values]
    elif unique_values.issubset({-1, 1}):
        spins = [int(value) for value in values]
    else:
        raise RuntimeError(
            "External MQLib output used unsupported assignment values: "
            f"{sorted(unique_values)}"
        )

    return {
        int(node): int(spin)
        for node, spin in zip(ordered_nodes, spins, strict=False)
    }


def external_burer_monteiro_maxcut(
    graph: nx.Graph,
    *,
    seed: int = 42,
    command: Sequence[str] | str | None = None,
    heuristic: str = "BURER2002",
    runtime_seconds: float | None = None,
) -> dict[int, int]:
    if graph.number_of_nodes() == 0:
        return {}

    resolved_command = _resolve_mqlib_command(command)
    runtime_limit = (
        float(runtime_seconds)
        if runtime_seconds is not None
        else max(0.25, min(5.0, 0.02 * max(1, graph.number_of_edges())))
    )
    if runtime_limit <= 0.0:
        raise ValueError("bm_external_runtime_seconds must be positive.")

    temp_handle = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".mqlib",
        delete=False,
        encoding="ascii",
        newline="\n",
    )
    temp_path = Path(temp_handle.name)
    temp_handle.close()

    try:
        ordered_nodes = _write_mqlib_maxcut_instance(graph, temp_path)
        process = subprocess.run(
            [
                *resolved_command,
                "-fM",
                str(temp_path),
                "-h",
                str(heuristic),
                "-r",
                f"{runtime_limit:.6f}",
                "-ps",
                "-s",
                str(int(seed) % 65536),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=max(10.0, runtime_limit + 10.0),
        )
    finally:
        temp_path.unlink(missing_ok=True)

    if process.returncode != 0:
        stderr = (process.stderr or "").strip()
        stdout = (process.stdout or "").strip()
        details = stderr or stdout or f"exit code {process.returncode}"
        raise RuntimeError(f"External MQLib solve failed: {details}")

    return _parse_mqlib_solution(process.stdout or "", ordered_nodes)


def goemans_williamson_maxcut(
    graph: nx.Graph,
    *,
    seed: int = 42,
    num_cuts: int = 32,
) -> dict[int, int]:
    if graph.number_of_nodes() == 0:
        return {}

    try:
        from qiskit_optimization.algorithms import GoemansWilliamsonOptimizer
        from qiskit_optimization.applications import Maxcut
    except ImportError as exc:
        raise ImportError(
            "Goemans-Williamson requires qiskit-optimization. "
            "Install it with `python -m pip install qiskit-optimization`."
        ) from exc

    ordered_nodes = tuple(sorted(int(node) for node in graph.nodes()))
    node_to_index = {node: index for index, node in enumerate(ordered_nodes)}

    indexed_graph = nx.Graph()
    indexed_graph.add_nodes_from(range(len(ordered_nodes)))
    for u, v, data in graph.edges(data=True):
        indexed_graph.add_edge(
            node_to_index[int(u)],
            node_to_index[int(v)],
            weight=float(data.get("weight", 1.0)),
        )

    optimizer = GoemansWilliamsonOptimizer(
        num_cuts=max(1, int(num_cuts)),
        seed=int(seed),
    )
    problem = Maxcut(indexed_graph).to_quadratic_program()
    result = optimizer.solve(problem)
    bits = np.asarray(result.x, dtype=int).reshape(-1)

    if len(bits) != len(ordered_nodes):
        raise RuntimeError(
            "Goemans-Williamson returned an assignment with the wrong size: "
            f"expected {len(ordered_nodes)}, got {len(bits)}."
        )

    return {
        node: 1 if int(bits[index]) == 0 else -1
        for index, node in enumerate(ordered_nodes)
    }


def burer_monteiro_maxcut(
    graph: nx.Graph,
    *,
    seed: int = 42,
    restarts: int = 24,
    rank: int | None = None,
    iterations: int = 64,
    rounding_trials: int = 64,
    initial_assignment: dict[int, int] | None = None,
    show_progress: bool = False,
) -> dict[int, int]:
    if graph.number_of_nodes() == 0:
        return {}

    ordered_nodes = tuple(sorted(int(node) for node in graph.nodes()))
    node_to_index = {node: index for index, node in enumerate(ordered_nodes)}
    node_count = len(ordered_nodes)
    embedding_dim = max(2, min(node_count, int(rank) if rank is not None else 8))
    neighbor_data = [
        [
            (node_to_index[int(neighbor)], float(data.get("weight", 1.0)))
            for neighbor, data in graph[node].items()
            if abs(float(data.get("weight", 1.0))) > EPS
        ]
        for node in ordered_nodes
    ]

    rng = np.random.default_rng(seed)
    best_assignment: dict[int, int] | None = None
    best_value = float("-inf")

    if initial_assignment is not None:
        seeded_assignment = {
            int(node): 1 if int(initial_assignment.get(int(node), 1)) > 0 else -1
            for node in ordered_nodes
        }
        seeded_value = weighted_cut_value(graph, seeded_assignment)
        best_assignment = seeded_assignment
        best_value = seeded_value

    restart_total = max(1, int(restarts))
    for restart_index in _progress_iterable(
        range(restart_total),
        enabled=show_progress,
        total=restart_total,
        desc="BM restarts",
        leave=False,
    ):
        factors = _random_unit_vectors(node_count, embedding_dim, rng)
        if initial_assignment is not None and restart_index == 0:
            seeded = np.zeros((node_count, embedding_dim), dtype=float)
            seeded[:, 0] = [float(initial_assignment.get(node, 1)) for node in ordered_nodes]
            factors = seeded + 0.05 * _random_unit_vectors(node_count, embedding_dim, rng)
            norms = np.linalg.norm(factors, axis=1)
            zero_mask = norms <= EPS
            if np.any(zero_mask):
                factors[zero_mask] = _random_unit_vectors(int(np.count_nonzero(zero_mask)), embedding_dim, rng)
                norms = np.linalg.norm(factors, axis=1)
            factors = factors / norms[:, None]

        for _ in range(max(1, int(iterations))):
            for row_index in rng.permutation(node_count):
                local_field = np.zeros(embedding_dim, dtype=float)
                for neighbor_index, weight in neighbor_data[row_index]:
                    local_field += weight * factors[neighbor_index]
                norm = float(np.linalg.norm(local_field))
                if norm <= EPS:
                    factors[row_index] = _random_unit_vectors(1, embedding_dim, rng)[0]
                    continue
                # Coordinate ascent for Max-Cut: align opposite to the weighted local field.
                factors[row_index] = -local_field / norm

        deterministic_directions = [factors[:, 0]]
        for direction in deterministic_directions:
            spins = np.where(direction >= 0.0, 1, -1)
            candidate = {
                node: int(spins[index])
                for index, node in enumerate(ordered_nodes)
            }
            candidate_value = weighted_cut_value(graph, candidate)
            if candidate_value > best_value + EPS:
                best_assignment = candidate
                best_value = candidate_value

        for _ in range(max(1, int(rounding_trials))):
            direction = _random_unit_vectors(1, embedding_dim, rng)[0]
            projections = factors @ direction
            spins = np.where(projections >= 0.0, 1, -1)
            candidate = {
                node: int(spins[index])
                for index, node in enumerate(ordered_nodes)
            }
            candidate_value = weighted_cut_value(graph, candidate)
            if candidate_value > best_value + EPS:
                best_assignment = candidate
                best_value = candidate_value

    if best_assignment is None:
        raise RuntimeError("Unable to generate a Burer-Monteiro assignment.")
    return best_assignment


def solve_graph_classically(
    graph: nx.Graph,
    *,
    solver: str,
    seed: int,
    restarts: int,
    initial_assignment: dict[int, int] | None = None,
    gw_num_cuts: int = 32,
    bm_backend: str = "auto",
    bm_rank: int | None = None,
    bm_iterations: int = 64,
    bm_rounding_trials: int = 64,
    bm_external_command: Sequence[str] | str | None = None,
    bm_external_heuristic: str = "BURER2002",
    bm_external_runtime_seconds: float | None = None,
    show_progress: bool = False,
) -> dict[int, int]:
    """Dispatch to the requested classical Max-Cut heuristic and return a spin assignment."""
    normalized_solver = solver.strip().lower()
    if normalized_solver == "greedy":
        return multi_start_greedy_maxcut(
            graph,
            seed=seed,
            restarts=restarts,
            initial_assignment=initial_assignment,
            show_progress=show_progress,
        )
    if normalized_solver in {"gw", "goemans_williamson"}:
        return goemans_williamson_maxcut(
            graph,
            seed=seed,
            num_cuts=gw_num_cuts,
        )
    if normalized_solver == "bm":
        normalized_backend = bm_backend.strip().lower()
        if normalized_backend not in {"auto", "external", "internal"}:
            raise ValueError(f"Unknown BM backend: {bm_backend}")

        if normalized_backend in {"auto", "external"}:
            try:
                return external_burer_monteiro_maxcut(
                    graph,
                    seed=seed,
                    command=bm_external_command,
                    heuristic=bm_external_heuristic,
                    runtime_seconds=bm_external_runtime_seconds,
                )
            except FileNotFoundError:
                if normalized_backend == "external" or _normalize_external_command(bm_external_command):
                    raise

        if normalized_backend == "external":
            raise RuntimeError("External BM backend was requested but could not be used.")

        return burer_monteiro_maxcut(
            graph,
            seed=seed,
            restarts=restarts,
            rank=bm_rank,
            iterations=bm_iterations,
            rounding_trials=bm_rounding_trials,
            initial_assignment=initial_assignment,
            show_progress=show_progress,
        )
    raise ValueError(f"Unknown classical solver: {solver}")


def _split_block_fallback(nodes: Iterable[int]) -> tuple[set[int], set[int]]:
    ordered = sorted(int(node) for node in nodes)
    midpoint = max(1, len(ordered) // 2)
    return set(ordered[:midpoint]), set(ordered[midpoint:])


def recursive_partition_blocks(
    graph: nx.Graph,
    *,
    max_block_size: int = 8,
    seed: int = 42,
) -> tuple[tuple[int, ...], ...]:
    if max_block_size < 2:
        raise ValueError("max_block_size must be at least 2.")

    rng = np.random.default_rng(seed)
    pending = [set(int(node) for node in graph.nodes())]
    blocks: list[tuple[int, ...]] = []

    while pending:
        block = pending.pop()
        if len(block) <= max_block_size:
            blocks.append(tuple(sorted(block)))
            continue

        subgraph = graph.subgraph(block).copy()
        try:
            left, right = nx.community.kernighan_lin_bisection(
                subgraph,
                weight="weight",
                seed=int(rng.integers(0, 2**31 - 1)),
            )
            left = set(int(node) for node in left)
            right = set(int(node) for node in right)
            if not left or not right:
                raise ValueError("Empty block produced.")
        except Exception:
            left, right = _split_block_fallback(block)

        if not left or not right:
            blocks.append(tuple(sorted(block)))
            continue

        pending.extend([left, right])

    blocks.sort(key=lambda block: (len(block), block[0]))
    return tuple(blocks)


def _light_cone_nodes(
    graph: nx.Graph,
    seed_nodes: Sequence[int],
    *,
    hops: int,
    max_nodes: int | None,
) -> tuple[int, ...]:
    ordered_seed_nodes = tuple(sorted(set(int(node) for node in seed_nodes)))
    if not ordered_seed_nodes:
        return ()

    cutoff = max(1, int(hops))
    distances: dict[int, int] = {}

    for seed in ordered_seed_nodes:
        for node, distance in nx.single_source_shortest_path_length(
            graph,
            seed,
            cutoff=cutoff,
        ).items():
            node_id = int(node)
            local_distance = int(distance)
            previous = distances.get(node_id)
            if previous is None or local_distance < previous:
                distances[node_id] = local_distance

    candidates = set(distances)
    if max_nodes is None or len(candidates) <= int(max_nodes):
        return tuple(sorted(candidates))

    endpoints = set(ordered_seed_nodes)
    node_limit = max(2, int(max_nodes))
    ordered = sorted(
        candidates,
        key=lambda node: (
            0 if node in endpoints else 1,
            distances.get(node, cutoff + 1),
            node,
        ),
    )

    keep = set(ordered[:node_limit])
    keep.update(endpoints)
    return tuple(sorted(keep))


def build_standard_qaoa_preconditioned_graph(
    graph: nx.Graph,
    *,
    qc_name: str | None = None,
    quantum_processor_id: str | None = None,
    max_block_size: int = 8,
    light_cone_hops: int = 1,
    width: int = 9,
    shots: int = 128,
    seed: int = 42,
    max_light_cone_size: int | None = None,
    min_abs_weight: float = 1e-6,
    show_progress: bool = True,
) -> PreconditionedGraphBuild:
    """Build a surrogate graph whose weights come from local QAOA light-cone correlations."""
    if max_block_size < 2:
        raise ValueError("max_block_size must be at least 2.")
    cone_radius = max(1, int(light_cone_hops))
    pair_distance_cutoff = 2 * cone_radius
    light_cone_evaluations = 0
    max_observed_light_cone = 0
    weight_floor = max(float(min_abs_weight), EPS)
    pair_distances = {
        int(node): {int(neighbor): int(distance) for neighbor, distance in distances.items()}
        for node, distances in nx.all_pairs_shortest_path_length(
            graph,
            cutoff=pair_distance_cutoff,
        )
    }
    cone_pairs: dict[tuple[int, ...], list[tuple[int, int]]] = {}
    candidate_edges: set[tuple[int, int]] = set()
    preconditioned_graph = nx.Graph()
    preconditioned_graph.add_nodes_from(int(node) for node in graph.nodes())

    for node in sorted(int(node) for node in graph.nodes()):
        for neighbor, distance in sorted(pair_distances.get(node, {}).items()):
            if neighbor <= node or distance == 0 or distance > pair_distance_cutoff:
                continue

            pair = (node, neighbor)
            candidate_edges.add(pair)
            cone_nodes = _light_cone_nodes(
                graph,
                pair,
                hops=cone_radius,
                max_nodes=max_light_cone_size,
            )
            if len(cone_nodes) < 2:
                continue
            cone_pairs.setdefault(cone_nodes, []).append(pair)

    evaluation_cones = tuple(
        cone_nodes
        for cone_nodes, _ in sorted(
            cone_pairs.items(),
            key=lambda item: (len(item[0]), item[0]),
        )
    )

    _progress_message(
        (
            "[preconditioned] "
            f"Preparing {len(evaluation_cones)} light-cone QAOA evaluations "
            f"from {len(candidate_edges)} candidate node pairs."
        ),
        enabled=show_progress,
    )

    enumerated_cones = enumerate(evaluation_cones, start=1)
    for cone_index, cone_nodes in _progress_iterable(
        enumerated_cones,
        enabled=show_progress,
        total=len(evaluation_cones),
        desc="Preconditioning cones",
        leave=False,
    ):
        _progress_message(
            (
                "[preconditioned] "
                f"Evaluating cone {cone_index}/{len(evaluation_cones)} "
                f"(nodes={len(cone_nodes)}, pairs={len(cone_pairs[cone_nodes])})."
            ),
            enabled=show_progress,
        )
        cone_problem = build_region_problem(graph, cone_nodes)
        cone_run = run_qaoa_landscape(
            cone_problem.qaoa_graph,
            width=width,
            shots=shots,
            layers=cone_radius,
            seed=seed,
            qc_name=qc_name,
            quantum_processor_id=quantum_processor_id,
            show_progress=False,
        )
        correlations = estimate_correlations(cone_run["best_samples"])
        local_index = {node: index for index, node in enumerate(cone_problem.node_order)}

        max_observed_light_cone = max(max_observed_light_cone, len(cone_nodes))
        light_cone_evaluations += 1

        # Eq. (2) in the paper replaces W with the off-diagonal correlation matrix:
        # Z_ij^(p) = -<Z_i Z_j> for i != j.
        for left, right in cone_pairs[cone_nodes]:
            correlation = float(correlations[local_index[left], local_index[right]])
            weight = -float(np.clip(correlation, -1.0, 1.0))
            if abs(weight) <= weight_floor:
                continue
            preconditioned_graph.add_edge(int(left), int(right), weight=weight)

    _progress_message(
        (
            "[preconditioned] "
            f"Finished building surrogate graph with {preconditioned_graph.number_of_edges()} "
            f"weighted edges from {light_cone_evaluations} cone evaluations."
        ),
        enabled=show_progress,
    )

    return PreconditionedGraphBuild(
        graph=preconditioned_graph,
        blocks=evaluation_cones,
        candidate_edges=len(candidate_edges),
        correlated_edges=preconditioned_graph.number_of_edges(),
        light_cone_evaluations=light_cone_evaluations,
        max_light_cone_size=max_observed_light_cone,
    )


def export_weighted_graph_to_csv(
    graph: nx.Graph,
    output_csv_path: str | Path,
) -> Path:
    output_path = Path(output_csv_path)
    edge_rows = [
        {
            "node_1": int(u),
            "node_2": int(v),
            "weight": float(data.get("weight", 1.0)),
        }
        for u, v, data in sorted(
            graph.edges(data=True),
            key=lambda edge: (int(edge[0]), int(edge[1])),
        )
    ]
    frame = pd.DataFrame(edge_rows, columns=["node_1", "node_2", "weight"])
    frame.to_csv(output_path, index=False)
    return output_path


def build_and_export_preconditioned_region_graph(
    graph: nx.Graph,
    selected_nodes: Sequence[int] | str,
    *,
    output_csv_path: str | Path = "preconditioned_region_graph.csv",
    qc_name: str | None = None,
    quantum_processor_id: str | None = None,
    max_block_size: int = 8,
    light_cone_hops: int = 1,
    width: int = 9,
    shots: int = 128,
    seed: int = 42,
    max_light_cone_size: int | None = None,
    min_abs_weight: float = 1e-6,
    show_progress: bool = True,
) -> nx.Graph:
    """Build a region-level surrogate graph and write its weighted edge list to CSV."""
    _progress_message(
        "[preconditioned] Building and exporting preconditioned graph only (no classical solve).",
        enabled=show_progress,
    )
    region_problem = build_region_problem(graph, selected_nodes)
    build = build_standard_qaoa_preconditioned_graph(
        region_problem.original_graph,
        qc_name=qc_name,
        quantum_processor_id=quantum_processor_id,
        max_block_size=max_block_size,
        light_cone_hops=light_cone_hops,
        width=width,
        shots=shots,
        seed=seed,
        max_light_cone_size=max_light_cone_size,
        min_abs_weight=min_abs_weight,
        show_progress=show_progress,
    )
    exported_path = export_weighted_graph_to_csv(build.graph, output_csv_path)
    _progress_message(
        f"[preconditioned] Exported surrogate graph to {exported_path.resolve()}.",
        enabled=show_progress,
    )
    return build.graph


def _partitions_from_assignment(assignment: dict[int, int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    left = tuple(sorted(node for node, spin in assignment.items() if int(spin) > 0))
    right = tuple(sorted(node for node, spin in assignment.items() if int(spin) <= 0))
    return left, right


def solve_region_with_qaoa(
    region_problem: RegionProblem,
    *,
    qc_name: str | None = None,
    quantum_processor_id: str | None = None,
    layers: int = 1,
    width: int = 21,
    shots: int = 250,
    seed: int = 42,
    polish_restarts: int = 24,
    show_progress: bool = True,
) -> SolveResult:
    """Solve a small region directly with QAOA, then polish the sampled assignment classically."""
    qaoa_run = run_qaoa_landscape(
        region_problem.qaoa_graph,
        width=width,
        shots=shots,
        layers=layers,
        seed=seed,
        qc_name=qc_name,
        quantum_processor_id=quantum_processor_id,
        show_progress=show_progress,
    )
    sampled_assignment = assignment_from_bitstring(
        region_problem.node_order,
        qaoa_run["best_bitstring"],
    )
    polished_assignment = multi_start_greedy_maxcut(
        region_problem.original_graph,
        seed=seed,
        restarts=polish_restarts,
        initial_assignment=sampled_assignment,
    )
    left_partition, right_partition = _partitions_from_assignment(polished_assignment)

    return SolveResult(
        method="qaoa",
        qc_name=qaoa_run["qc_name"],
        selected_nodes=region_problem.node_order,
        assignment=polished_assignment,
        cut_value=weighted_cut_value(region_problem.original_graph, polished_assignment),
        left_partition=left_partition,
        right_partition=right_partition,
        sample_cut_value=float(qaoa_run["best_cut_value"]),
        mean_objective=float(qaoa_run["best_mean_objective"]),
        beta=qaoa_run["best_beta"],
        gamma=qaoa_run["best_gamma"],
        landscape=qaoa_run["landscape"],
        beta_values=qaoa_run["angle_values"],
        gamma_values=qaoa_run["angle_values"],
    )


def solve_region_with_bm(
    region_problem: RegionProblem,
    *,
    seed: int = 42,
    restarts: int = 24,
    bm_backend: str = "auto",
    bm_rank: int | None = None,
    bm_iterations: int = 64,
    bm_rounding_trials: int = 64,
    bm_external_command: Sequence[str] | str | None = None,
    bm_external_heuristic: str = "BURER2002",
    bm_external_runtime_seconds: float | None = None,
    show_progress: bool = True,
) -> SolveResult:
    """Solve a region with the Burer-Monteiro path and a final greedy local search."""
    assignment = solve_graph_classically(
        region_problem.original_graph,
        solver="bm",
        seed=seed,
        restarts=restarts,
        bm_backend=bm_backend,
        bm_rank=bm_rank,
        bm_iterations=bm_iterations,
        bm_rounding_trials=bm_rounding_trials,
        bm_external_command=bm_external_command,
        bm_external_heuristic=bm_external_heuristic,
        bm_external_runtime_seconds=bm_external_runtime_seconds,
        show_progress=show_progress,
    )
    assignment = greedy_local_search(region_problem.original_graph, assignment)
    left_partition, right_partition = _partitions_from_assignment(assignment)

    return SolveResult(
        method="bm",
        qc_name="classical",
        selected_nodes=region_problem.node_order,
        assignment=assignment,
        cut_value=weighted_cut_value(region_problem.original_graph, assignment),
        left_partition=left_partition,
        right_partition=right_partition,
    )


def solve_region_with_goemans_williamson(
    region_problem: RegionProblem,
    *,
    seed: int = 42,
    gw_num_cuts: int = 32,
    show_progress: bool = True,
) -> SolveResult:
    """Solve a region with Goemans-Williamson and then apply greedy local refinement."""
    assignment = solve_graph_classically(
        region_problem.original_graph,
        solver="goemans_williamson",
        seed=seed,
        restarts=1,
        gw_num_cuts=gw_num_cuts,
        show_progress=show_progress,
    )
    assignment = greedy_local_search(region_problem.original_graph, assignment)
    left_partition, right_partition = _partitions_from_assignment(assignment)

    return SolveResult(
        method="goemans_williamson",
        qc_name="classical",
        selected_nodes=region_problem.node_order,
        assignment=assignment,
        cut_value=weighted_cut_value(region_problem.original_graph, assignment),
        left_partition=left_partition,
        right_partition=right_partition,
    )


def solve_region_with_preconditioning(
    region_problem: RegionProblem,
    *,
    solver: str = "goemans_williamson",
    qc_name: str | None = None,
    quantum_processor_id: str | None = None,
    max_block_size: int = 8,
    light_cone_hops: int = 1,
    width: int = 9,
    shots: int = 128,
    seed: int = 42,
    restarts: int = 48,
    gw_num_cuts: int = 32,
    bm_backend: str = "auto",
    bm_rank: int | None = None,
    bm_iterations: int = 64,
    bm_rounding_trials: int = 64,
    bm_external_command: Sequence[str] | str | None = None,
    bm_external_heuristic: str = "BURER2002",
    bm_external_runtime_seconds: float | None = None,
    max_light_cone_size: int | None = None,
    min_abs_weight: float = 1e-6,
    show_progress: bool = True,
) -> SolveResult:
    """Build a QAOA-derived surrogate graph, solve it classically, then score on the original graph."""
    build = build_standard_qaoa_preconditioned_graph(
        region_problem.original_graph,
        qc_name=qc_name,
        quantum_processor_id=quantum_processor_id,
        max_block_size=max_block_size,
        light_cone_hops=light_cone_hops,
        width=width,
        shots=shots,
        seed=seed,
        max_light_cone_size=max_light_cone_size,
        min_abs_weight=min_abs_weight,
        show_progress=show_progress,
    )
    normalized_solver = solver.strip().lower()
    _progress_message(
        (
            "[preconditioned] "
            f"Solving surrogate graph with {build.graph.number_of_edges()} edges using "
            f"{normalized_solver}."
        ),
        enabled=show_progress,
    )
    assignment = solve_graph_classically(
        build.graph,
        solver=normalized_solver,
        seed=seed,
        restarts=restarts,
        gw_num_cuts=gw_num_cuts,
        bm_backend=bm_backend,
        bm_rank=bm_rank,
        bm_iterations=bm_iterations,
        bm_rounding_trials=bm_rounding_trials,
        bm_external_command=bm_external_command,
        bm_external_heuristic=bm_external_heuristic,
        bm_external_runtime_seconds=bm_external_runtime_seconds,
        show_progress=show_progress,
    )
    assignment = greedy_local_search(region_problem.original_graph, assignment)
    left_partition, right_partition = _partitions_from_assignment(assignment)
    _progress_message(
        (
            "[preconditioned] "
            f"Finished classical solve on surrogate graph. Final cut value on original graph: "
            f"{weighted_cut_value(region_problem.original_graph, assignment):.6f}."
        ),
        enabled=show_progress,
    )

    return SolveResult(
        method=f"preconditioned_{normalized_solver}",
        qc_name=_resolve_qc_name(
            len(region_problem.node_order),
            qc_name,
            quantum_processor_id,
        ),
        selected_nodes=region_problem.node_order,
        assignment=assignment,
        cut_value=weighted_cut_value(region_problem.original_graph, assignment),
        left_partition=left_partition,
        right_partition=right_partition,
        preconditioned_graph=build.graph,
        blocks=build.blocks,
    )


def solve_selected_region(
    graph: nx.Graph,
    selected_nodes: Sequence[int] | str,
    *,
    method: str = "auto",
    qc_name: str | None = None,
    quantum_processor_id: str | None = None,
    qaoa_node_limit: int = DEFAULT_QAOA_NODE_LIMIT,
    qaoa_layers: int = 1,
    width: int = 21,
    shots: int = 250,
    seed: int = 42,
    polish_restarts: int = 24,
    gw_num_cuts: int = 32,
    bm_restarts: int = 24,
    bm_backend: str = "auto",
    bm_rank: int | None = None,
    bm_iterations: int = 64,
    bm_rounding_trials: int = 64,
    bm_external_command: Sequence[str] | str | None = None,
    bm_external_heuristic: str = "BURER2002",
    bm_external_runtime_seconds: float | None = None,
    precondition_width: int = 9,
    precondition_shots: int = 128,
    precondition_block_size: int = 8,
    precondition_light_cone_hops: int = 1,
    precondition_restarts: int = 48,
    precondition_max_light_cone_size: int | None = None,
    precondition_min_abs_weight: float = 1e-6,
    precondition_output_csv_path: str | Path = "preconditioned_region_graph.csv",
    show_progress: bool = True,
) -> SolveResult | nx.Graph:
    """Choose a solve path for one region and execute it end-to-end."""
    region_problem = build_region_problem(graph, selected_nodes)
    normalized_method = method.strip().lower()

    if normalized_method == "auto":
        normalized_method = (
            "qaoa"
            if len(region_problem.node_order) <= int(qaoa_node_limit)
            else "preconditioned_goemans_williamson"
        )

    if normalized_method == "qaoa":
        return solve_region_with_qaoa(
            region_problem,
            qc_name=qc_name,
            quantum_processor_id=quantum_processor_id,
            layers=qaoa_layers,
            width=width,
            shots=shots,
            seed=seed,
            polish_restarts=polish_restarts,
            show_progress=show_progress,
        )

    if normalized_method in {"gw", "goemans_williamson"}:
        return solve_region_with_goemans_williamson(
            region_problem,
            seed=seed,
            gw_num_cuts=gw_num_cuts,
            show_progress=show_progress,
        )

    if normalized_method == "bm":
        return solve_region_with_bm(
            region_problem,
            seed=seed,
            restarts=bm_restarts,
            bm_backend=bm_backend,
            bm_rank=bm_rank,
            bm_iterations=bm_iterations,
            bm_rounding_trials=bm_rounding_trials,
            bm_external_command=bm_external_command,
            bm_external_heuristic=bm_external_heuristic,
            bm_external_runtime_seconds=bm_external_runtime_seconds,
            show_progress=show_progress,
        )

    if normalized_method == "preconditioned":
        return build_and_export_preconditioned_region_graph(
            graph,
            region_problem.node_order,
            output_csv_path=precondition_output_csv_path,
            qc_name=qc_name,
            quantum_processor_id=quantum_processor_id,
            max_block_size=precondition_block_size,
            light_cone_hops=precondition_light_cone_hops,
            width=precondition_width,
            shots=precondition_shots,
            seed=seed,
            max_light_cone_size=precondition_max_light_cone_size,
            min_abs_weight=precondition_min_abs_weight,
            show_progress=show_progress,
        )

    if normalized_method in {
        "preconditioned_qaoa",
        "preconditioned_gw",
        "preconditioned_goemans_williamson",
    }:
        return solve_region_with_preconditioning(
            region_problem,
            solver="goemans_williamson",
            qc_name=qc_name,
            quantum_processor_id=quantum_processor_id,
            max_block_size=precondition_block_size,
            light_cone_hops=precondition_light_cone_hops,
            width=precondition_width,
            shots=precondition_shots,
            seed=seed,
            restarts=precondition_restarts,
            gw_num_cuts=gw_num_cuts,
            bm_backend=bm_backend,
            bm_rank=bm_rank,
            bm_iterations=bm_iterations,
            bm_rounding_trials=bm_rounding_trials,
            bm_external_command=bm_external_command,
            bm_external_heuristic=bm_external_heuristic,
            bm_external_runtime_seconds=bm_external_runtime_seconds,
            max_light_cone_size=precondition_max_light_cone_size,
            min_abs_weight=precondition_min_abs_weight,
            show_progress=show_progress,
        )

    if normalized_method == "preconditioned_bm":
        return solve_region_with_preconditioning(
            region_problem,
            solver="bm",
            qc_name=qc_name,
            quantum_processor_id=quantum_processor_id,
            max_block_size=precondition_block_size,
            light_cone_hops=precondition_light_cone_hops,
            width=precondition_width,
            shots=precondition_shots,
            seed=seed,
            restarts=precondition_restarts,
            gw_num_cuts=gw_num_cuts,
            bm_backend=bm_backend,
            bm_rank=bm_rank,
            bm_iterations=bm_iterations,
            bm_rounding_trials=bm_rounding_trials,
            bm_external_command=bm_external_command,
            bm_external_heuristic=bm_external_heuristic,
            bm_external_runtime_seconds=bm_external_runtime_seconds,
            max_light_cone_size=precondition_max_light_cone_size,
            min_abs_weight=precondition_min_abs_weight,
            show_progress=show_progress,
        )

    if normalized_method == "preconditioned_greedy":
        return solve_region_with_preconditioning(
            region_problem,
            solver="greedy",
            qc_name=qc_name,
            quantum_processor_id=quantum_processor_id,
            max_block_size=precondition_block_size,
            light_cone_hops=precondition_light_cone_hops,
            width=precondition_width,
            shots=precondition_shots,
            seed=seed,
            restarts=precondition_restarts,
            gw_num_cuts=gw_num_cuts,
            bm_backend=bm_backend,
            bm_rank=bm_rank,
            bm_iterations=bm_iterations,
            bm_rounding_trials=bm_rounding_trials,
            bm_external_command=bm_external_command,
            bm_external_heuristic=bm_external_heuristic,
            bm_external_runtime_seconds=bm_external_runtime_seconds,
            max_light_cone_size=precondition_max_light_cone_size,
            min_abs_weight=precondition_min_abs_weight,
            show_progress=show_progress,
        )

    raise ValueError(f"Unknown method: {method}")


def plot_landscape(
    landscape: np.ndarray | None,
    *,
    beta_values: Sequence[float],
    gamma_values: Sequence[float],
    title: str | None = None,
) -> None:
    """Render the single-layer QAOA landscape returned by :func:`run_qaoa_landscape`."""
    import matplotlib.pyplot as plt

    if landscape is None or np.asarray(landscape).ndim != 2:
        raise ValueError("plot_landscape only supports the single-layer 2D QAOA sweep.")

    beta_axis = np.asarray(beta_values, dtype=float)
    gamma_axis = np.asarray(gamma_values, dtype=float)
    gamma_index, beta_index = np.unravel_index(int(np.argmax(landscape)), landscape.shape)

    plt.figure(figsize=(8, 5))
    plt.imshow(
        landscape,
        extent=[beta_axis[0], beta_axis[-1], gamma_axis[-1], gamma_axis[0]],
        aspect="auto",
    )
    plt.plot(beta_axis[beta_index], gamma_axis[gamma_index], "ro")
    plt.colorbar(label="Expected cut value")
    plt.xlabel("beta (radians)")
    plt.ylabel("gamma (radians)")
    plt.title(title or "Weighted Max-Cut QAOA Landscape")
    plt.show()
