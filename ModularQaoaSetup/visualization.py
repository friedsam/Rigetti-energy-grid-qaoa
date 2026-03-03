from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def _ensure_parent_dir(save: Optional[Union[str, Path]]) -> Optional[Path]:
    if save is None:
        return None
    p = Path(save)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def spring_pos(G: nx.Graph, seed: int = 7) -> Dict[int, np.ndarray]:
    # A stable spring layout heuristic
    k = 1 / np.sqrt(max(1, G.number_of_nodes()))
    return nx.spring_layout(G, seed=seed, k=k)


def draw_graph(
    G: nx.Graph,
    *,
    title: str = "",
    pos: Optional[Dict[int, np.ndarray]] = None,
    node_color=None,
    node_size: int = 80,
    with_labels: bool = False,
    seed: int = 7,
    save: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> Dict[int, np.ndarray]:
    """
    Generic graph draw helper.
    Returns the position dict used (so you can reuse positions across plots).
    """
    if pos is None:
        pos = spring_pos(G, seed=seed)

    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=node_size, alpha=0.9)
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.35)

    if with_labels:
        nx.draw_networkx_labels(G, pos, font_size=7)

    plt.title(title)
    plt.axis("off")

    out = _ensure_parent_dir(save)
    if out is not None:
        plt.savefig(out, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    return pos


def plot_boundary_nodes(
    G: nx.Graph,
    boundary_nodes: Iterable[int],
    *,
    title: str = "Boundary nodes (red) vs interior (blue)",
    pos: Optional[Dict[int, np.ndarray]] = None,
    seed: int = 7,
    save: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> Dict[int, np.ndarray]:
    boundary = set(int(n) for n in boundary_nodes)
    colors = ["red" if int(n) in boundary else "steelblue" for n in G.nodes()]
    return draw_graph(G, title=title, pos=pos, node_color=colors, seed=seed, save=save, show=show)


def plot_regions(
    G: nx.Graph,
    region_blocks: Sequence[Sequence[int]],
    *,
    title: str = "Regions (color by region id)",
    pos: Optional[Dict[int, np.ndarray]] = None,
    seed: int = 7,
    save: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> Dict[int, np.ndarray]:
    node_to_region: Dict[int, int] = {}
    for rid, block in enumerate(region_blocks):
        for n in block:
            node_to_region[int(n)] = rid

    # Ensure all nodes are assigned
    missing = [int(n) for n in G.nodes() if int(n) not in node_to_region]
    if missing:
        raise ValueError(f"plot_regions: {len(missing)} nodes not assigned to any region block.")

    region_ids = np.array([node_to_region[int(n)] for n in G.nodes()], dtype=int)
    return draw_graph(G, title=title, pos=pos, node_color=region_ids, seed=seed, save=save, show=show)


def plot_coarse_graph(
    coarse_graph: nx.Graph,
    *,
    title: str = "Coarse graph (regions as nodes)",
    node_size: int = 250,
    seed: int = 7,
    save: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> Dict[int, np.ndarray]:
    return draw_graph(coarse_graph, title=title, node_size=node_size, seed=seed, save=save, show=show)