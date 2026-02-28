from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict

import networkx as nx

from .qaoa_solvers.modular import ModularSolveConfig, solve_graph_modular
from .resource_accounting import CreditCheckpoint, ResourceSnapshot, ResourceTracker, tracking_resources


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    partition_strategy: str
    region_optimizer: str
    boundary_optimizer: str
    wall_time_s: float
    cpu_time_s: float
    weighted_cut: float
    ising_energy: float
    rounds_run: int
    resources: ResourceSnapshot
    credit_checkpoints: tuple[CreditCheckpoint, ...]

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "name": self.name,
            "partition_strategy": self.partition_strategy,
            "region_optimizer": self.region_optimizer,
            "boundary_optimizer": self.boundary_optimizer,
            "wall_time_s": self.wall_time_s,
            "cpu_time_s": self.cpu_time_s,
            "weighted_cut": self.weighted_cut,
            "ising_energy": self.ising_energy,
            "rounds_run": self.rounds_run,
        }
        payload.update(self.resources.__dict__)
        return payload


def benchmark_modular_run(
    graph: nx.Graph,
    name: str,
    config: ModularSolveConfig | None = None,
    *,
    separate_timing_pass: bool = True,
) -> BenchmarkResult:
    config = ModularSolveConfig() if config is None else config
    tracker = ResourceTracker()

    if separate_timing_pass:
        start_wall = time.perf_counter()
        start_cpu = time.process_time()
        timed_result = solve_graph_modular(graph, name=name, config=config)
        wall_time_s = time.perf_counter() - start_wall
        cpu_time_s = time.process_time() - start_cpu
    else:
        timed_result = None
        wall_time_s = None
        cpu_time_s = None

    with tracking_resources(tracker):
        if separate_timing_pass:
            counted_result = solve_graph_modular(graph, name=name, config=config)
        else:
            start_wall = time.perf_counter()
            start_cpu = time.process_time()
            counted_result = solve_graph_modular(graph, name=name, config=config)
            wall_time_s = time.perf_counter() - start_wall
            cpu_time_s = time.process_time() - start_cpu

    result = counted_result if timed_result is None else timed_result
    if wall_time_s is None or cpu_time_s is None:
        raise RuntimeError("Benchmark timing was not captured.")

    return BenchmarkResult(
        name=name,
        partition_strategy=config.partition_strategy,
        region_optimizer=config.region_optimizer,
        boundary_optimizer=config.boundary_optimizer,
        wall_time_s=wall_time_s,
        cpu_time_s=cpu_time_s,
        weighted_cut=result.cut_value,
        ising_energy=result.ising_energy,
        rounds_run=result.rounds_run,
        resources=tracker.snapshot,
        credit_checkpoints=tuple(tracker.checkpoints),
    )
