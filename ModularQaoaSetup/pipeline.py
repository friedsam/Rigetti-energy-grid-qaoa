from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, Sequence

import networkx as nx

from .resource_accounting import record_checkpoint


Assignment = Dict[int, int]
BlockUpdater = Callable[[nx.Graph, Sequence[int], Assignment, int], None]
RefinementStep = Callable[[nx.Graph, Assignment, int], Assignment]
ObjectiveFunction = Callable[[nx.Graph, Assignment], float]


@dataclass(frozen=True)
class PipelineResult:
    assignment: Assignment
    rounds_run: int


def run_block_coordinate_descent(
    graph: nx.Graph,
    assignment: Assignment,
    *,
    region_blocks: Sequence[Sequence[int]],
    boundary_blocks: Sequence[Sequence[int]],
    rounds: int,
    seed: int,
    refine_assignment: RefinementStep,
    objective_value: ObjectiveFunction,
    region_updater: BlockUpdater | None = None,
    boundary_updater: BlockUpdater | None = None,
    shuffle_region_blocks: bool = False,
    shuffle_boundary_blocks: bool = False,
    rollback_on_regression: bool = True,
    stop_on_stall: bool = True,
) -> PipelineResult:
    current_assignment = assignment.copy()
    region_blocks = tuple(tuple(int(node) for node in block) for block in region_blocks)
    boundary_blocks = tuple(tuple(int(node) for node in block) for block in boundary_blocks)
    rng = random.Random(seed)
    rounds_run = 0

    for round_index in range(max(1, rounds)):
        previous_assignment = current_assignment.copy()
        before_value = objective_value(graph, current_assignment)

        region_order = list(range(len(region_blocks)))
        if shuffle_region_blocks:
            rng.shuffle(region_order)
        for order_index, block_index in enumerate(region_order):
            if region_updater is None:
                break
            region_updater(
                graph,
                region_blocks[block_index],
                current_assignment,
                seed + round_index * 101 + order_index,
            )
            record_checkpoint(
                objective_value(graph, current_assignment),
                label="region_block",
            )

        boundary_order = list(range(len(boundary_blocks)))
        if shuffle_boundary_blocks:
            rng.shuffle(boundary_order)
        for order_index, block_index in enumerate(boundary_order):
            if boundary_updater is None:
                break
            boundary_updater(
                graph,
                boundary_blocks[block_index],
                current_assignment,
                seed + 5000 + round_index * 101 + order_index,
            )
            record_checkpoint(
                objective_value(graph, current_assignment),
                label="boundary_block",
            )

        current_assignment = refine_assignment(graph, current_assignment, seed + round_index + 1)
        record_checkpoint(
            objective_value(graph, current_assignment),
            label="refine",
        )
        after_value = objective_value(graph, current_assignment)
        rounds_run = round_index + 1

        if rollback_on_regression and after_value + 1e-9 < before_value:
            current_assignment = previous_assignment
            break

        if stop_on_stall and after_value <= before_value + 1e-9:
            break

    return PipelineResult(assignment=current_assignment, rounds_run=rounds_run)
