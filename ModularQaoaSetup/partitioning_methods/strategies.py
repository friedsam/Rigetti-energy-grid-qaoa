from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import networkx as nx
import numpy as np


@dataclass
class RegionLayout:
    regions: List[tuple[int, ...]]
    node_to_region: Dict[int, int]
    coarse_graph: nx.Graph
    boundary_nodes: tuple[int, ...]


@dataclass(frozen=True)
class PartitionSchedule:
    local_block_limit: int
    layout: RegionLayout | None
    region_blocks: tuple[tuple[int, ...], ...]
    boundary_blocks: tuple[tuple[int, ...], ...]


def _fallback_bisect(node_group: Sequence[int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    ordered_group = tuple(sorted(int(node) for node in node_group))
    midpoint = max(1, len(ordered_group) // 2)
    left = ordered_group[:midpoint]
    right = ordered_group[midpoint:]
    if not right:
        return left, tuple(reversed(left[-1:]))
    return left, right


def _split_with_kernighan_lin(
    subgraph: nx.Graph,
    ordered_group: Sequence[int],
    seed: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    try:
        left, right = nx.algorithms.community.kernighan_lin_bisection(
            subgraph,
            weight="weight",
            seed=seed,
        )
        left = tuple(sorted(int(node) for node in left))
        right = tuple(sorted(int(node) for node in right))
    except Exception:
        left, right = _fallback_bisect(ordered_group)
    if not left or not right:
        left, right = _fallback_bisect(ordered_group)
    return left, right


def _split_with_spectral(
    subgraph: nx.Graph,
    ordered_group: Sequence[int],
    seed: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    del seed
    if len(ordered_group) <= 2:
        return _fallback_bisect(ordered_group)

    try:
        laplacian = nx.laplacian_matrix(subgraph, nodelist=ordered_group, weight="weight").astype(float)
        dense = laplacian.toarray()
        if np.allclose(dense, 0.0):
            raise ValueError("Degenerate Laplacian")
        eigenvalues, eigenvectors = np.linalg.eigh(dense)
        vector_index = 1 if len(eigenvalues) > 1 else 0
        fiedler = np.real(eigenvectors[:, vector_index])
        if np.max(np.abs(fiedler)) < 1e-12:
            raise ValueError("Degenerate Fiedler vector")
        order = np.argsort(fiedler)
        midpoint = max(1, len(order) // 2)
        left = tuple(sorted(int(ordered_group[index]) for index in order[:midpoint]))
        right = tuple(sorted(int(ordered_group[index]) for index in order[midpoint:]))
    except Exception:
        left, right = _fallback_bisect(ordered_group)

    if not left or not right:
        left, right = _fallback_bisect(ordered_group)
    return left, right


def _split_with_random_balance(
    subgraph: nx.Graph,
    ordered_group: Sequence[int],
    seed: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    del subgraph
    rng = random.Random(seed)
    nodes = list(int(node) for node in ordered_group)
    rng.shuffle(nodes)
    midpoint = max(1, len(nodes) // 2)
    left = tuple(sorted(nodes[:midpoint]))
    right = tuple(sorted(nodes[midpoint:]))
    if not left or not right:
        left, right = _fallback_bisect(ordered_group)
    return left, right


def _split_with_girvan_newman(
    subgraph: nx.Graph,
    ordered_group: Sequence[int],
    seed: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    del seed
    try:
        communities = next(nx.algorithms.community.girvan_newman(subgraph))
        left, right = communities
        left = tuple(sorted(int(node) for node in left))
        right = tuple(sorted(int(node) for node in right))
    except Exception:
        left, right = _fallback_bisect(ordered_group)
    if not left or not right:
        left, right = _fallback_bisect(ordered_group)
    return left, right


def _recursive_partition_with_splitter(
    graph: nx.Graph,
    max_block_size: int,
    seed: int,
    splitter,
) -> List[tuple[int, ...]]:
    if max_block_size < 2:
        return [(int(node),) for node in sorted(graph.nodes())]

    blocks: List[tuple[int, ...]] = []

    def split(node_group: Iterable[int], local_seed: int) -> None:
        ordered_group = tuple(sorted(set(int(node) for node in node_group)))
        if not ordered_group:
            return
        if len(ordered_group) <= max_block_size:
            blocks.append(ordered_group)
            return

        subgraph = graph.subgraph(ordered_group).copy()
        components = list(nx.connected_components(subgraph))
        if len(components) > 1:
            for index, component in enumerate(components):
                split(component, local_seed + index + 1)
            return

        left, right = splitter(subgraph, ordered_group, local_seed)
        if not left or not right:
            left, right = _fallback_bisect(ordered_group)
        if set(left) == set(ordered_group) or set(right) == set(ordered_group):
            left, right = _fallback_bisect(ordered_group)

        split(left, local_seed + 1)
        split(right, local_seed + 2)

    split(graph.nodes(), seed)
    return sorted(blocks, key=lambda block: (len(block), block))


def _recursive_partition_with_splitter_cycle(
    graph: nx.Graph,
    max_block_size: int,
    seed: int,
    splitters: Sequence,
) -> List[tuple[int, ...]]:
    if max_block_size < 2:
        return [(int(node),) for node in sorted(graph.nodes())]
    if not splitters:
        raise ValueError("At least one splitter is required.")

    blocks: List[tuple[int, ...]] = []

    def split(node_group: Iterable[int], local_seed: int, depth: int) -> None:
        ordered_group = tuple(sorted(set(int(node) for node in node_group)))
        if not ordered_group:
            return
        if len(ordered_group) <= max_block_size:
            blocks.append(ordered_group)
            return

        subgraph = graph.subgraph(ordered_group).copy()
        components = list(nx.connected_components(subgraph))
        if len(components) > 1:
            for index, component in enumerate(components):
                split(component, local_seed + index + 1, depth + 1)
            return

        splitter = splitters[depth % len(splitters)]
        left, right = splitter(subgraph, ordered_group, local_seed)
        if not left or not right:
            left, right = _fallback_bisect(ordered_group)
        if set(left) == set(ordered_group) or set(right) == set(ordered_group):
            left, right = _fallback_bisect(ordered_group)

        split(left, local_seed + 1, depth + 1)
        split(right, local_seed + 2, depth + 1)

    split(graph.nodes(), seed, 0)
    return sorted(blocks, key=lambda block: (len(block), block))


def recursive_partition(
    graph: nx.Graph,
    max_block_size: int,
    seed: int,
) -> List[tuple[int, ...]]:
    return _recursive_partition_with_splitter(
        graph,
        max_block_size=max_block_size,
        seed=seed,
        splitter=_split_with_kernighan_lin,
    )


def spectral_partition(
    graph: nx.Graph,
    max_block_size: int,
    seed: int,
) -> List[tuple[int, ...]]:
    return _recursive_partition_with_splitter(
        graph,
        max_block_size=max_block_size,
        seed=seed,
        splitter=_split_with_spectral,
    )


def random_balanced_partition(
    graph: nx.Graph,
    max_block_size: int,
    seed: int,
) -> List[tuple[int, ...]]:
    return _recursive_partition_with_splitter(
        graph,
        max_block_size=max_block_size,
        seed=seed,
        splitter=_split_with_random_balance,
    )


def girvan_newman_partition(
    graph: nx.Graph,
    max_block_size: int,
    seed: int,
) -> List[tuple[int, ...]]:
    return _recursive_partition_with_splitter(
        graph,
        max_block_size=max_block_size,
        seed=seed,
        splitter=_split_with_girvan_newman,
    )


def recursive_spectral_kl_partition(
    graph: nx.Graph,
    max_block_size: int,
    seed: int,
) -> List[tuple[int, ...]]:
    return _recursive_partition_with_splitter_cycle(
        graph,
        max_block_size=max_block_size,
        seed=seed,
        splitters=(_split_with_spectral, _split_with_kernighan_lin),
    )


def recursive_kl_girvan_partition(
    graph: nx.Graph,
    max_block_size: int,
    seed: int,
) -> List[tuple[int, ...]]:
    return _recursive_partition_with_splitter_cycle(
        graph,
        max_block_size=max_block_size,
        seed=seed,
        splitters=(_split_with_kernighan_lin, _split_with_girvan_newman),
    )


def recursive_spectral_girvan_partition(
    graph: nx.Graph,
    max_block_size: int,
    seed: int,
) -> List[tuple[int, ...]]:
    return _recursive_partition_with_splitter_cycle(
        graph,
        max_block_size=max_block_size,
        seed=seed,
        splitters=(_split_with_spectral, _split_with_girvan_newman),
    )


def partition_into_regions(
    graph: nx.Graph,
    max_region_size: int,
    seed: int,
) -> List[tuple[int, ...]]:
    if graph.number_of_nodes() <= max_region_size:
        return [tuple(sorted(int(node) for node in graph.nodes()))]

    try:
        communities = list(
            nx.algorithms.community.greedy_modularity_communities(
                graph,
                weight="weight",
            )
        )
    except Exception:
        communities = [set(graph.nodes())]

    if not communities:
        communities = [set(graph.nodes())]

    regions: List[tuple[int, ...]] = []
    ordered_communities = sorted(
        communities,
        key=lambda community: (len(community), tuple(sorted(int(node) for node in community))),
    )

    for index, community in enumerate(ordered_communities):
        community_nodes = tuple(sorted(int(node) for node in community))
        if len(community_nodes) <= max_region_size:
            regions.append(community_nodes)
            continue

        subgraph = graph.subgraph(community_nodes).copy()
        regions.extend(
            recursive_partition(
                subgraph,
                max_block_size=max_region_size,
                seed=seed + index * 17 + 1,
            )
        )

    covered = {node for region in regions for node in region}
    missing = sorted(int(node) for node in set(graph.nodes()).difference(covered))
    for node in missing:
        regions.append((node,))

    return sorted(regions, key=lambda region: (len(region), region))


def build_region_layout(
    graph: nx.Graph,
    max_region_size: int,
    seed: int,
) -> RegionLayout:
    regions = partition_into_regions(graph, max_region_size=max_region_size, seed=seed)
    node_to_region: Dict[int, int] = {}
    for region_index, region in enumerate(regions):
        for node in region:
            node_to_region[int(node)] = region_index

    coarse_graph = nx.Graph()
    for region_index, region in enumerate(regions):
        coarse_graph.add_node(region_index, members=region, size=len(region))

    for u, v, data in graph.edges(data=True):
        region_u = node_to_region[int(u)]
        region_v = node_to_region[int(v)]
        if region_u == region_v:
            continue

        weight = float(data["weight"])
        if coarse_graph.has_edge(region_u, region_v):
            coarse_graph[region_u][region_v]["weight"] += weight
        else:
            coarse_graph.add_edge(region_u, region_v, weight=weight)

    boundary_nodes = tuple(
        sorted(
            int(node)
            for node in graph.nodes()
            if any(
                node_to_region[int(neighbor)] != node_to_region[int(node)]
                for neighbor in graph.neighbors(node)
            )
        )
    )

    return RegionLayout(
        regions=regions,
        node_to_region=node_to_region,
        coarse_graph=coarse_graph,
        boundary_nodes=boundary_nodes,
    )


def boundary_blocks_from_layout(
    graph: nx.Graph,
    layout: RegionLayout,
    max_block_size: int,
    seed: int,
    subpartitioner=None,
) -> List[tuple[int, ...]]:
    if subpartitioner is None:
        subpartitioner = recursive_partition

    boundary_set = set(layout.boundary_nodes)
    if not boundary_set:
        return []

    boundary_graph = nx.Graph()
    boundary_graph.add_nodes_from(boundary_set)

    for u, v, data in graph.edges(data=True):
        if int(u) not in boundary_set or int(v) not in boundary_set:
            continue
        if layout.node_to_region[int(u)] == layout.node_to_region[int(v)]:
            continue
        boundary_graph.add_edge(int(u), int(v), weight=float(data["weight"]))

    blocks: List[tuple[int, ...]] = []
    for component_index, component in enumerate(nx.connected_components(boundary_graph)):
        component_nodes = tuple(sorted(int(node) for node in component))
        if len(component_nodes) <= max_block_size:
            blocks.append(component_nodes)
            continue

        component_graph = boundary_graph.subgraph(component_nodes).copy()
        blocks.extend(
            subpartitioner(
                component_graph,
                max_block_size=max_block_size,
                seed=seed + component_index + 1,
            )
        )

    return sorted(blocks, key=lambda block: (len(block), block))


def build_multilevel_partition_schedule(
    graph: nx.Graph,
    max_block_size: int,
    seed: int,
) -> PartitionSchedule:
    return _build_multilevel_partition_schedule_with_subpartitioner(
        graph,
        max_block_size=max_block_size,
        seed=seed,
        subpartitioner=recursive_partition,
    )


def _build_multilevel_partition_schedule_with_subpartitioner(
    graph: nx.Graph,
    max_block_size: int,
    seed: int,
    subpartitioner,
) -> PartitionSchedule:
    local_block_limit = max(2, max_block_size)
    layout = build_region_layout(graph, max_region_size=local_block_limit, seed=seed)

    region_blocks: List[tuple[int, ...]] = []
    for region_index, region in enumerate(layout.regions):
        if len(region) <= local_block_limit:
            region_blocks.append(region)
            continue

        region_graph = graph.subgraph(region).copy()
        region_blocks.extend(
            subpartitioner(
                region_graph,
                max_block_size=local_block_limit,
                seed=seed + region_index * 31 + 1,
            )
        )

    boundary_blocks = boundary_blocks_from_layout(
        graph,
        layout=layout,
        max_block_size=local_block_limit,
        seed=seed + 1000,
        subpartitioner=subpartitioner,
    )

    return PartitionSchedule(
        local_block_limit=local_block_limit,
        layout=layout,
        region_blocks=tuple(region_blocks),
        boundary_blocks=tuple(boundary_blocks),
    )


def build_recursive_partition_schedule(
    graph: nx.Graph,
    max_block_size: int,
    seed: int,
) -> PartitionSchedule:
    local_block_limit = max(2, max_block_size)
    blocks = recursive_partition(graph, max_block_size=local_block_limit, seed=seed)
    return PartitionSchedule(
        local_block_limit=local_block_limit,
        layout=None,
        region_blocks=tuple(blocks),
        boundary_blocks=(),
    )


def build_spectral_partition_schedule(
    graph: nx.Graph,
    max_block_size: int,
    seed: int,
) -> PartitionSchedule:
    local_block_limit = max(2, max_block_size)
    blocks = spectral_partition(graph, max_block_size=local_block_limit, seed=seed)
    return PartitionSchedule(
        local_block_limit=local_block_limit,
        layout=None,
        region_blocks=tuple(blocks),
        boundary_blocks=(),
    )


def build_random_balanced_partition_schedule(
    graph: nx.Graph,
    max_block_size: int,
    seed: int,
) -> PartitionSchedule:
    local_block_limit = max(2, max_block_size)
    blocks = random_balanced_partition(graph, max_block_size=local_block_limit, seed=seed)
    return PartitionSchedule(
        local_block_limit=local_block_limit,
        layout=None,
        region_blocks=tuple(blocks),
        boundary_blocks=(),
    )


def build_girvan_newman_partition_schedule(
    graph: nx.Graph,
    max_block_size: int,
    seed: int,
) -> PartitionSchedule:
    local_block_limit = max(2, max_block_size)
    blocks = girvan_newman_partition(graph, max_block_size=local_block_limit, seed=seed)
    return PartitionSchedule(
        local_block_limit=local_block_limit,
        layout=None,
        region_blocks=tuple(blocks),
        boundary_blocks=(),
    )


def build_multilevel_spectral_partition_schedule(
    graph: nx.Graph,
    max_block_size: int,
    seed: int,
) -> PartitionSchedule:
    return _build_multilevel_partition_schedule_with_subpartitioner(
        graph,
        max_block_size=max_block_size,
        seed=seed,
        subpartitioner=spectral_partition,
    )


def build_multilevel_girvan_newman_partition_schedule(
    graph: nx.Graph,
    max_block_size: int,
    seed: int,
) -> PartitionSchedule:
    return _build_multilevel_partition_schedule_with_subpartitioner(
        graph,
        max_block_size=max_block_size,
        seed=seed,
        subpartitioner=girvan_newman_partition,
    )


def build_recursive_spectral_kl_partition_schedule(
    graph: nx.Graph,
    max_block_size: int,
    seed: int,
) -> PartitionSchedule:
    local_block_limit = max(2, max_block_size)
    blocks = recursive_spectral_kl_partition(graph, max_block_size=local_block_limit, seed=seed)
    return PartitionSchedule(
        local_block_limit=local_block_limit,
        layout=None,
        region_blocks=tuple(blocks),
        boundary_blocks=(),
    )


def build_recursive_kl_girvan_partition_schedule(
    graph: nx.Graph,
    max_block_size: int,
    seed: int,
) -> PartitionSchedule:
    local_block_limit = max(2, max_block_size)
    blocks = recursive_kl_girvan_partition(graph, max_block_size=local_block_limit, seed=seed)
    return PartitionSchedule(
        local_block_limit=local_block_limit,
        layout=None,
        region_blocks=tuple(blocks),
        boundary_blocks=(),
    )


def build_recursive_spectral_girvan_partition_schedule(
    graph: nx.Graph,
    max_block_size: int,
    seed: int,
) -> PartitionSchedule:
    local_block_limit = max(2, max_block_size)
    blocks = recursive_spectral_girvan_partition(graph, max_block_size=local_block_limit, seed=seed)
    return PartitionSchedule(
        local_block_limit=local_block_limit,
        layout=None,
        region_blocks=tuple(blocks),
        boundary_blocks=(),
    )


def build_partition_schedule(
    graph: nx.Graph,
    max_block_size: int,
    seed: int,
    strategy: str = "multilevel",
) -> PartitionSchedule:
    normalized = strategy.strip().lower()
    if normalized == "multilevel":
        return build_multilevel_partition_schedule(graph, max_block_size=max_block_size, seed=seed)
    if normalized == "recursive":
        return build_recursive_partition_schedule(graph, max_block_size=max_block_size, seed=seed)
    if normalized == "spectral":
        return build_spectral_partition_schedule(graph, max_block_size=max_block_size, seed=seed)
    if normalized == "random_balanced":
        return build_random_balanced_partition_schedule(graph, max_block_size=max_block_size, seed=seed)
    if normalized == "girvan_newman":
        return build_girvan_newman_partition_schedule(graph, max_block_size=max_block_size, seed=seed)
    if normalized == "multilevel_spectral":
        return build_multilevel_spectral_partition_schedule(graph, max_block_size=max_block_size, seed=seed)
    if normalized == "multilevel_girvan_newman":
        return build_multilevel_girvan_newman_partition_schedule(graph, max_block_size=max_block_size, seed=seed)
    if normalized == "recursive_spectral_kl":
        return build_recursive_spectral_kl_partition_schedule(graph, max_block_size=max_block_size, seed=seed)
    if normalized == "recursive_kl_girvan":
        return build_recursive_kl_girvan_partition_schedule(graph, max_block_size=max_block_size, seed=seed)
    if normalized == "recursive_spectral_girvan":
        return build_recursive_spectral_girvan_partition_schedule(graph, max_block_size=max_block_size, seed=seed)
    raise ValueError(f"Unknown partition strategy: {strategy}")


def available_partition_strategies() -> tuple[str, ...]:
    return (
        "multilevel",
        "recursive",
        "spectral",
        "random_balanced",
        "girvan_newman",
        "multilevel_spectral",
        "multilevel_girvan_newman",
        "recursive_spectral_kl",
        "recursive_kl_girvan",
        "recursive_spectral_girvan",
    )
