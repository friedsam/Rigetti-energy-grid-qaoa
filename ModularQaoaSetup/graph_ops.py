from __future__ import annotations

import math
import random
from typing import Dict, Sequence

import networkx as nx
import numpy as np
from networkx.algorithms.approximation.maxcut import one_exchange

from .resource_accounting import record_local_search_pass


def weighted_cut_value(graph: nx.Graph, assignment: Dict[int, int]) -> float:
    total = 0.0
    for u, v, data in graph.edges(data=True):
        if assignment[u] != assignment[v]:
            total += float(data["weight"])
    return total


def ising_energy(graph: nx.Graph, assignment: Dict[int, int]) -> float:
    total = 0.0
    for u, v, data in graph.edges(data=True):
        total += float(data["weight"]) * assignment[u] * assignment[v]
    return total


def initial_assignment(graph: nx.Graph, seed: int) -> Dict[int, int]:
    try:
        _, partition = one_exchange(graph, weight="weight", seed=seed)
        left, right = partition
    except Exception:
        nodes = list(graph.nodes())
        midpoint = len(nodes) // 2
        left = set(nodes[:midpoint])
        right = set(nodes[midpoint:])

    left = set(left)
    right = set(right)
    assignment = {node: (1 if node in left else -1) for node in graph.nodes()}
    if not right:
        for index, node in enumerate(sorted(graph.nodes())):
            assignment[node] = 1 if index % 2 == 0 else -1
    return assignment


def greedy_refine(
    graph: nx.Graph,
    assignment: Dict[int, int],
    max_passes: int = 25,
    seed: int = 0,
) -> Dict[int, int]:
    rng = random.Random(seed)
    nodes = list(graph.nodes())
    for _ in range(max_passes):
        rng.shuffle(nodes)
        improved = False
        edge_touches = 0
        flips = 0
        for node in nodes:
            neighbor_sum = 0.0
            for neighbor, data in graph[node].items():
                edge_touches += 1
                neighbor_sum += float(data["weight"]) * assignment[neighbor]
            delta_energy = -2.0 * assignment[node] * neighbor_sum
            if delta_energy < -1e-12:
                assignment[node] *= -1
                improved = True
                flips += 1
        record_local_search_pass(
            node_visits=len(nodes),
            edge_touches=edge_touches,
            flips=flips,
        )
        if not improved:
            break
    return assignment


def simulated_annealing_refine(
    graph: nx.Graph,
    assignment: Dict[int, int],
    *,
    blocks: Sequence[Sequence[int]] | None = None,
    temperatures: int = 12,
    sweeps_per_temperature: int = 2,
    initial_temperature: float | None = None,
    final_temperature: float = 0.05,
    seed: int = 0,
) -> Dict[int, int]:
    rng = random.Random(seed)
    ordered_blocks = (
        tuple(tuple(int(node) for node in block) for block in blocks)
        if blocks is not None
        else (tuple(int(node) for node in graph.nodes()),)
    )
    if not ordered_blocks:
        return assignment

    if initial_temperature is None:
        total_abs_weight = sum(abs(float(data["weight"])) for _, _, data in graph.edges(data=True))
        average_abs_weight = total_abs_weight / max(1, graph.number_of_nodes())
        initial_temperature = max(0.5, average_abs_weight)

    start_temp = max(float(initial_temperature), 1e-6)
    end_temp = max(min(float(final_temperature), start_temp), 1e-6)
    num_temps = max(1, int(temperatures))

    if num_temps == 1:
        schedule = [start_temp]
    else:
        ratio = end_temp / start_temp
        schedule = [start_temp * (ratio ** (index / (num_temps - 1))) for index in range(num_temps)]

    for temperature in schedule:
        for _ in range(max(1, int(sweeps_per_temperature))):
            for block in ordered_blocks:
                if not block:
                    continue
                nodes = list(block)
                rng.shuffle(nodes)
                edge_touches = 0
                flips = 0
                for node in nodes:
                    neighbor_sum = 0.0
                    for neighbor, data in graph[node].items():
                        edge_touches += 1
                        neighbor_sum += float(data["weight"]) * assignment[neighbor]
                    delta_energy = -2.0 * assignment[node] * neighbor_sum
                    accept = delta_energy < -1e-12
                    if not accept and temperature > 1e-12:
                        probability = math.exp(-delta_energy / temperature)
                        accept = rng.random() < min(1.0, probability)
                    if accept:
                        assignment[node] *= -1
                        flips += 1
                record_local_search_pass(
                    node_visits=len(nodes),
                    edge_touches=edge_touches,
                    flips=flips,
                )

    return assignment


def burer_monteiro_refine(
    graph: nx.Graph,
    assignment: Dict[int, int],
    *,
    blocks: Sequence[Sequence[int]] | None = None,
    rank: int = 8,
    steps: int = 80,
    learning_rate: float = 0.2,
    rounding_trials: int = 32,
    seed: int = 0,
) -> Dict[int, int]:
    rng = np.random.default_rng(seed)
    ordered_blocks = (
        tuple(tuple(int(node) for node in block) for block in blocks)
        if blocks is not None
        else (tuple(int(node) for node in graph.nodes()),)
    )
    if not ordered_blocks:
        return assignment

    vector_rank = max(2, int(rank))
    num_steps = max(1, int(steps))
    base_learning_rate = max(1e-4, float(learning_rate))
    num_rounding_trials = max(1, int(rounding_trials))

    for raw_block in ordered_blocks:
        block = tuple(sorted(set(int(node) for node in raw_block)))
        if not block:
            continue

        if len(block) == 1:
            node = block[0]
            neighbor_sum = 0.0
            edge_touches = 0
            for neighbor, data in graph[node].items():
                edge_touches += 1
                neighbor_sum += float(data["weight"]) * assignment[int(neighbor)]
            previous = assignment[node]
            if neighbor_sum > 1e-12:
                assignment[node] = -1
            elif neighbor_sum < -1e-12:
                assignment[node] = 1
            flips = 1 if assignment[node] != previous else 0
            record_local_search_pass(
                node_visits=1,
                edge_touches=edge_touches,
                flips=flips,
            )
            continue

        block_size = len(block)
        node_to_index = {node: index for index, node in enumerate(block)}
        pair_weights = np.zeros((block_size, block_size), dtype=float)
        fields = np.zeros(block_size, dtype=float)
        edge_touches = 0

        for i, node in enumerate(block):
            for neighbor, data in graph[node].items():
                edge_touches += 1
                weight = float(data["weight"])
                neighbor_index = node_to_index.get(int(neighbor))
                if neighbor_index is None:
                    fields[i] += weight * assignment[int(neighbor)]
                    continue
                if i < neighbor_index:
                    pair_weights[i, neighbor_index] = weight
                    pair_weights[neighbor_index, i] = weight

        pair_terms = int(np.count_nonzero(np.triu(np.abs(pair_weights) > 1e-12, k=1)))
        state = rng.normal(size=(block_size, vector_rank))
        state[:, 0] += np.array([assignment[node] for node in block], dtype=float)
        norms = np.linalg.norm(state, axis=1, keepdims=True)
        state /= np.maximum(norms, 1e-12)

        for step_index in range(num_steps):
            gradients = pair_weights @ state
            gradients[:, 0] += fields
            step_size = base_learning_rate / math.sqrt(step_index + 1.0)
            state = state - step_size * gradients
            norms = np.linalg.norm(state, axis=1, keepdims=True)
            state /= np.maximum(norms, 1e-12)
            record_local_search_pass(
                node_visits=block_size,
                edge_touches=edge_touches + pair_terms,
                flips=0,
            )

        def local_energy(spins: np.ndarray) -> float:
            quadratic = 0.5 * float(spins @ pair_weights @ spins)
            linear = float(spins @ fields)
            return quadratic + linear

        current_spins = np.array([assignment[node] for node in block], dtype=int)
        best_spins = current_spins.copy()
        best_energy = local_energy(best_spins.astype(float))

        axis_spins = np.where(state[:, 0] >= 0.0, 1, -1).astype(int)
        axis_energy = local_energy(axis_spins.astype(float))
        if axis_energy < best_energy:
            best_spins = axis_spins
            best_energy = axis_energy

        for _ in range(num_rounding_trials):
            hyperplane = rng.normal(size=vector_rank)
            anchor_sign = 1 if hyperplane[0] >= 0.0 else -1
            projections = state @ hyperplane
            rounded = np.where(projections * anchor_sign >= 0.0, 1, -1).astype(int)
            energy = local_energy(rounded.astype(float))
            if energy < best_energy:
                best_spins = rounded
                best_energy = energy

        flips = 0
        for node, spin in zip(block, best_spins):
            updated_spin = int(spin)
            if assignment[node] != updated_spin:
                flips += 1
            assignment[node] = updated_spin

        record_local_search_pass(
            node_visits=block_size * max(1, num_rounding_trials + 1),
            edge_touches=(edge_touches + pair_terms) * max(1, num_rounding_trials + 1),
            flips=flips,
        )

    return assignment
