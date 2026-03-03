"""Partition the challenge graph into MITES-style regions for downstream solvers.

The helpers in this module load the weighted edge list, derive size-bounded
regions using community-detection heuristics, score each region for likely
quantum benefit, and emit CSV artifacts that make the partition easy to inspect
or reuse in later hardware-execution steps.
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

DEFAULT_PROBLEM_B_PARQUET = "019c9083-9e69-72f0-b313-85026d9a88aa.parquet"
DEFAULT_NODE_MAP_CSV = "problem_b_region_node_map.csv"
DEFAULT_REGION_SUMMARY_CSV = "problem_b_region_summary.csv"
DEFAULT_RANDOM_STATE = 42
DEFAULT_MIN_REGION_NODES = 6
DEFAULT_MAX_REGION_NODES = 24
QAOA_NODE_LIMIT = 10


def resolve_problem_b_parquet(parquet_path: str | Path | None = None) -> Path:
    """Resolve the challenge parquet file, preferring an explicit user path."""
    if parquet_path is not None:
        candidate = Path(parquet_path)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Parquet file not found: {candidate}")

    default_path = Path(DEFAULT_PROBLEM_B_PARQUET)
    if default_path.exists():
        return default_path

    scored_candidates: list[tuple[int, int, Path]] = []
    for candidate in Path(".").glob("*.parquet"):
        frame = pd.read_parquet(candidate)
        nodes = set(frame["node_1"]).union(frame["node_2"])
        scored_candidates.append((len(nodes), len(frame), candidate))

    if not scored_candidates:
        raise FileNotFoundError("No parquet files were found in the current directory.")

    scored_candidates.sort(reverse=True)
    return scored_candidates[0][2]


def load_weighted_graph(parquet_path: str | Path | None = None) -> tuple[Path, pd.DataFrame, nx.Graph]:
    """Load the weighted challenge graph from parquet into pandas and NetworkX views."""
    path = resolve_problem_b_parquet(parquet_path)
    edge_frame = pd.read_parquet(path)

    graph = nx.Graph()
    for row in edge_frame.itertuples(index=False):
        graph.add_edge(int(row.node_1), int(row.node_2), weight=float(row.weight))

    return path, edge_frame, graph


def _louvain_seed_regions(graph: nx.Graph, random_state: int) -> list[set[int]]:
    if hasattr(nx.community, "louvain_communities"):
        communities = nx.community.louvain_communities(graph, weight="weight", seed=random_state)
    else:
        communities = nx.community.greedy_modularity_communities(graph, weight="weight")
    return [set(community) for community in communities]


def _split_large_region(
    graph: nx.Graph,
    region_nodes: set[int],
    max_region_nodes: int,
    min_region_nodes: int,
    random_state: int,
) -> list[set[int]]:
    pending = [set(region_nodes)]
    output: list[set[int]] = []

    while pending:
        current = pending.pop()
        if len(current) <= max_region_nodes:
            output.append(current)
            continue

        subgraph = graph.subgraph(current).copy()
        try:
            left, right = nx.community.kernighan_lin_bisection(
                subgraph,
                weight="weight",
                seed=random_state,
            )
        except Exception:
            output.append(current)
            continue

        left = set(left)
        right = set(right)
        if not left or not right:
            output.append(current)
            continue

        if min(len(left), len(right)) < min_region_nodes:
            output.append(current)
            continue

        pending.extend([left, right])

    return output


def _adjacent_cut_weight(graph: nx.Graph, left: set[int], right: set[int]) -> float | None:
    total = 0.0
    adjacent = False
    for node in left:
        for neighbor, data in graph[node].items():
            if neighbor in right:
                adjacent = True
                total += float(data.get("weight", 1.0))
    if not adjacent:
        return None
    return total


def _merge_small_regions(graph: nx.Graph, regions: list[set[int]], min_region_nodes: int) -> list[set[int]]:
    merged = [set(region) for region in regions if region]

    while len(merged) > 1:
        small_indexes = [index for index, region in enumerate(merged) if len(region) < min_region_nodes]
        if not small_indexes:
            break

        merge_choice: tuple[int, int] | None = None
        ordered_small_indexes = sorted(
            small_indexes,
            key=lambda index: (len(merged[index]), min(merged[index])),
        )

        for source_index in ordered_small_indexes:
            source_region = merged[source_index]
            best_index = None
            best_score = float("-inf")

            for target_index, target_region in enumerate(merged):
                if target_index == source_index:
                    continue

                score = _adjacent_cut_weight(graph, source_region, target_region)
                if score is None:
                    continue
                if score > best_score:
                    best_score = score
                    best_index = target_index

            if best_index is not None:
                merge_choice = (source_index, best_index)
                break

        if merge_choice is None:
            break

        source_index, best_index = merge_choice
        source_region = merged[source_index]
        merged[best_index] = merged[best_index].union(source_region)
        del merged[source_index]

    return merged


def partition_with_mites(
    graph: nx.Graph,
    min_region_nodes: int = DEFAULT_MIN_REGION_NODES,
    max_region_nodes: int = DEFAULT_MAX_REGION_NODES,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> list[set[int]]:
    """
    Build MITES-style regions via a METIS-like heuristic:
    1. Seed regions from weighted modularity communities.
    2. Split oversized regions with Kernighan-Lin bisection.
    3. Merge undersized regions into the strongest adjacent neighbor.
    """

    seeded_regions = _louvain_seed_regions(graph, random_state=random_state)

    size_balanced_regions: list[set[int]] = []
    for region in seeded_regions:
        size_balanced_regions.extend(
            _split_large_region(
                graph,
                region,
                max_region_nodes=max_region_nodes,
                min_region_nodes=min_region_nodes,
                random_state=random_state,
            )
        )

    final_regions = _merge_small_regions(graph, size_balanced_regions, min_region_nodes=min_region_nodes)
    final_regions.sort(key=lambda nodes: (min(nodes), len(nodes)))
    return final_regions


def _json_dump(value: object) -> str:
    return json.dumps(value, separators=(",", ":"))


def _unique_region_boundary_edges(graph: nx.Graph, region_nodes: set[int]) -> list[dict[str, object]]:
    region = set(region_nodes)
    boundary_edges: list[dict[str, object]] = []

    for node in sorted(region):
        for neighbor in sorted(graph.neighbors(node)):
            if neighbor in region:
                continue
            boundary_edges.append(
                {
                    "source": int(node),
                    "target": int(neighbor),
                    "weight": float(graph[node][neighbor].get("weight", 1.0)),
                }
            )

    return boundary_edges


def _region_internal_edges(graph: nx.Graph, region_nodes: set[int]) -> list[dict[str, object]]:
    subgraph = graph.subgraph(region_nodes)
    internal_edges: list[dict[str, object]] = []

    for left, right, data in sorted(subgraph.edges(data=True)):
        internal_edges.append(
            {
                "source": int(left),
                "target": int(right),
                "weight": float(data.get("weight", 1.0)),
            }
        )

    return internal_edges


def _minmax_normalize(values: list[float]) -> list[float]:
    if not values:
        return []

    array = np.asarray(values, dtype=float)
    low = float(array.min())
    high = float(array.max())
    if np.isclose(low, high):
        return [1.0 for _ in values]

    return ((array - low) / (high - low)).tolist()


def compute_region_records(
    graph: nx.Graph,
    regions: list[set[int]],
    random_state: int = DEFAULT_RANDOM_STATE,
) -> list[dict[str, object]]:
    """Build per-region metrics used to categorize classical, QAOA, and hybrid regions."""
    base_records: list[dict[str, object]] = []
    density_values: list[float] = []
    core_values: list[float] = []

    for index, region_nodes in enumerate(regions):
        ordered_nodes = sorted(region_nodes)
        subgraph = graph.subgraph(region_nodes).copy()
        node_count = subgraph.number_of_nodes()
        edge_count = subgraph.number_of_edges()
        boundary_edges = _unique_region_boundary_edges(graph, region_nodes)
        internal_edges = _region_internal_edges(graph, region_nodes)

        boundary_edge_count = len(boundary_edges)
        boundary_weight = sum(edge["weight"] for edge in boundary_edges)
        internal_weight = sum(edge["weight"] for edge in internal_edges)

        total_incident_edge_count = edge_count + boundary_edge_count
        bridge_frac = (
            boundary_edge_count / total_incident_edge_count if total_incident_edge_count else 0.0
        )

        if edge_count:
            internal_bridge_count = len(list(nx.bridges(subgraph)))
            cycle_ratio = (edge_count - internal_bridge_count) / edge_count
            core_number = max(nx.core_number(subgraph).values())
        else:
            cycle_ratio = 0.0
            core_number = 0

        clustering = nx.average_clustering(subgraph, weight="weight") if node_count > 1 else 0.0
        density = nx.density(subgraph) if node_count > 1 else 0.0
        core_norm_raw = core_number / max(node_count - 1, 1)

        base_records.append(
            {
                "region_id": f"R{index:02d}",
                "region_nodes": ordered_nodes,
                "node_count": node_count,
                "edge_count": edge_count,
                "boundary_edge_count": boundary_edge_count,
                "boundary_weight": boundary_weight,
                "internal_weight": internal_weight,
                "cycle_ratio": cycle_ratio,
                "clustering": clustering,
                "density": density,
                "core_norm_raw": core_norm_raw,
                "bridge_frac": bridge_frac,
                "internal_edges": internal_edges,
                "boundary_edges": boundary_edges,
            }
        )
        density_values.append(density)
        core_values.append(core_norm_raw)

    density_norms = _minmax_normalize(density_values)
    core_norms = _minmax_normalize(core_values)

    frustration_scores: list[float] = []
    for record, density_norm, core_norm in zip(base_records, density_norms, core_norms):
        base_score = (
            0.62 * float(record["cycle_ratio"])
            + 0.18 * float(record["clustering"])
            + 0.15 * density_norm
            + 0.05 * core_norm
        )
        frustration_score = 0.5 * base_score * max(0.0, 1.0 - 0.85 * float(record["bridge_frac"]))

        record["density_norm"] = density_norm
        record["core_norm"] = core_norm
        record["frustration_score"] = frustration_score
        frustration_scores.append(frustration_score)

    rounded_scores = np.round(np.asarray(frustration_scores, dtype=float), 12)
    if len(base_records) >= 2 and np.unique(rounded_scores).size > 1:
        model = KMeans(n_clusters=2, random_state=random_state, n_init=10)
        labels = model.fit_predict(rounded_scores.reshape(-1, 1))
        high_label = int(np.argmax(model.cluster_centers_.ravel()))
    else:
        labels = np.zeros(len(base_records), dtype=int)
        high_label = 0

    for record, label in zip(base_records, labels):
        frustration_band = "high" if int(label) == high_label else "low"
        if len(base_records) == 1:
            frustration_band = "high" if float(record["frustration_score"]) > 0.0 else "low"

        if frustration_band == "low":
            region_category = "classical"
        elif int(record["node_count"]) > QAOA_NODE_LIMIT:
            region_category = "quantum preconditioning"
        else:
            region_category = "qaoa"

        record["frustration_band"] = frustration_band
        record["region_category"] = region_category
        record["region_nodes_json"] = _json_dump(record["region_nodes"])
        record["internal_edges_json"] = _json_dump(record["internal_edges"])
        record["boundary_edges_json"] = _json_dump(record["boundary_edges"])

    return base_records


def _region_summary_frame(region_records: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for record in region_records:
        rows.append(
            {
                "region_id": record["region_id"],
                "region_category": record["region_category"],
                "frustration_band": record["frustration_band"],
                "node_count": record["node_count"],
                "edge_count": record["edge_count"],
                "boundary_edge_count": record["boundary_edge_count"],
                "internal_weight": record["internal_weight"],
                "boundary_weight": record["boundary_weight"],
                "cycle_ratio": record["cycle_ratio"],
                "clustering": record["clustering"],
                "density": record["density"],
                "density_norm": record["density_norm"],
                "core_norm": record["core_norm"],
                "bridge_frac": record["bridge_frac"],
                "frustration_score": record["frustration_score"],
                "region_nodes_json": record["region_nodes_json"],
                "internal_edges_json": record["internal_edges_json"],
                "boundary_edges_json": record["boundary_edges_json"],
            }
        )
    return pd.DataFrame(rows)


def build_node_map(graph: nx.Graph, region_records: list[dict[str, object]]) -> pd.DataFrame:
    """Expand region metadata into a per-node lookup table with serialized edge context."""
    region_by_node: dict[int, dict[str, object]] = {}
    region_id_by_node: dict[int, str] = {}

    for record in region_records:
        for node in record["region_nodes"]:
            region_by_node[int(node)] = record
            region_id_by_node[int(node)] = str(record["region_id"])

    node_rows = []
    for node in sorted(graph.nodes()):
        record = region_by_node[int(node)]
        incident_edges = []
        intra_region_edges = []
        cross_region_edges = []

        for neighbor in sorted(graph.neighbors(node)):
            edge_payload = {
                "source": int(node),
                "target": int(neighbor),
                "weight": float(graph[node][neighbor].get("weight", 1.0)),
                "target_region": region_id_by_node[int(neighbor)],
            }
            incident_edges.append(edge_payload)
            if edge_payload["target_region"] == record["region_id"]:
                intra_region_edges.append(edge_payload)
            else:
                cross_region_edges.append(edge_payload)

        node_rows.append(
            {
                "node_id": int(node),
                "region_id": record["region_id"],
                "region_category": record["region_category"],
                "frustration_band": record["frustration_band"],
                "region_node_count": record["node_count"],
                "region_edge_count": record["edge_count"],
                "region_frustration_score": record["frustration_score"],
                "region_nodes_json": record["region_nodes_json"],
                "incident_edges_json": _json_dump(incident_edges),
                "intra_region_edges_json": _json_dump(intra_region_edges),
                "cross_region_edges_json": _json_dump(cross_region_edges),
                "reconstruction_note": (
                    "Explode incident_edges_json and keep one copy of each undirected edge "
                    "by sorting (source,target)."
                ),
            }
        )

    return pd.DataFrame(node_rows)


def run_pipeline(
    parquet_path: str | Path | None = None,
    node_map_csv: str | Path = DEFAULT_NODE_MAP_CSV,
    region_summary_csv: str | Path = DEFAULT_REGION_SUMMARY_CSV,
    min_region_nodes: int = DEFAULT_MIN_REGION_NODES,
    max_region_nodes: int = DEFAULT_MAX_REGION_NODES,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full partitioning workflow and persist the summary and node-map CSVs."""
    _, _, graph = load_weighted_graph(parquet_path=parquet_path)
    regions = partition_with_mites(
        graph,
        min_region_nodes=min_region_nodes,
        max_region_nodes=max_region_nodes,
        random_state=random_state,
    )
    region_records = compute_region_records(graph, regions, random_state=random_state)

    region_summary = _region_summary_frame(region_records)
    node_map = build_node_map(graph, region_records)

    region_summary.to_csv(region_summary_csv, index=False)
    node_map.to_csv(node_map_csv, index=False)
    return region_summary, node_map


def main() -> None:
    """Run the default pipeline from the current working directory and print a short summary."""
    input_path = resolve_problem_b_parquet()
    region_summary, node_map = run_pipeline(parquet_path=input_path)

    print(f"Input parquet: {input_path}")
    print(f"Regions: {len(region_summary)}")
    print(
        region_summary[
            [
                "region_id",
                "node_count",
                "frustration_score",
                "frustration_band",
                "region_category",
            ]
        ].to_string(index=False)
    )
    print(f"\nNode map rows: {len(node_map)}")
    print(f"Wrote: {DEFAULT_REGION_SUMMARY_CSV}")
    print(f"Wrote: {DEFAULT_NODE_MAP_CSV}")


if __name__ == "__main__":
    main()
