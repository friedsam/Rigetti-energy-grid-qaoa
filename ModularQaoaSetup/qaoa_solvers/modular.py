"""Top-level orchestration for the modular QAOA and hybrid block solver.

This module turns a graph plus a high-level configuration into the partitioned
solve pipeline used by the rest of the package. It selects the partitioning
strategy, wires in the requested block updater, and routes whole-graph hybrid
preconditioning through the dedicated preconditioned solver path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Sequence

import networkx as nx

from ..graph_ops import greedy_refine, initial_assignment, ising_energy, weighted_cut_value
from ..partitioning_methods.strategies import PartitionSchedule, build_partition_schedule
from ..pipeline import BlockUpdater, run_block_coordinate_descent
from ..resource_accounting import record_checkpoint
from .hyperparameters import (
    LibraryQAOAOptimizerConfig,
    StandardQAOAOptimizerConfig,
    WarmStartQAOAOptimizerConfig,
)


@dataclass(frozen=True)
class ModularSolveConfig:
    """Configuration for partitioning, quantum subsolvers, and fallback heuristics."""
    partition_strategy: str = "multilevel"
    region_optimizer: str = "hybrid_preconditioned"
    boundary_optimizer: str = "hybrid_preconditioned"
    depth: int = 1
    max_block_size: int = 16
    rounds: int = 3
    restarts: int = 3
    maxiter: int = 35
    seed: int = 7
    warm_start_assignment: Dict[int, int] | None = None
    preconditioning_mode: Literal["bcd", "global_bm_step"] = "bcd"
    max_light_cone_size: int | None = None
    preconditioner_use_pcut: bool = False
    preconditioner_shots: int = 0
    error_model: Literal["ideal", "ankaa3", "ankaa3_hardware"] = "ideal"
    error_model_specs_path: str = "Ankaa-3_device_specs.csv"
    error_model_layout_seed: int = 0
    preconditioning_backend: Literal["simulated_annealing", "burer_monteiro", "greedy"] = "burer_monteiro"
    sa_temperatures: int = 12
    sa_sweeps_per_temperature: int = 2
    sa_initial_temperature: float | None = None
    sa_final_temperature: float = 0.05
    bm_rank: int = 6
    bm_steps: int = 24
    bm_learning_rate: float = 0.2
    bm_rounding_trials: int = 16
    standard_qaoa_optimizer: StandardQAOAOptimizerConfig = field(default_factory=StandardQAOAOptimizerConfig)
    library_qaoa_optimizer: LibraryQAOAOptimizerConfig = field(default_factory=LibraryQAOAOptimizerConfig)
    warm_start_qaoa_optimizer: WarmStartQAOAOptimizerConfig = field(default_factory=WarmStartQAOAOptimizerConfig)
    preconditioner_min_abs_weight: float = 1e-6
    exact_threshold: int = 21
    allow_exact_postcheck: bool = True

    def __post_init__(self) -> None:
        region_strategy = self.region_optimizer.strip().lower()
        boundary_strategy = self.boundary_optimizer.strip().lower()
        uses_preconditioned_solver = (
            region_strategy == "hybrid_preconditioned" or boundary_strategy == "hybrid_preconditioned"
        )
        if uses_preconditioned_solver and (
            region_strategy != "hybrid_preconditioned" or boundary_strategy != "hybrid_preconditioned"
        ):
            raise ValueError(
                "hybrid_preconditioned must be used for both region and boundary optimizers. "
                "It now builds a whole preconditioned graph before the classical solve."
            )


def available_partitioning_strategies() -> tuple[str, ...]:
    """Return the partitioning strategy names accepted by ``partition_strategy``."""
    from ..partitioning_methods.strategies import available_partition_strategies

    return available_partition_strategies()


def available_optimization_strategies() -> tuple[str, ...]:
    """Return the block-optimizer names accepted by region and boundary settings."""
    return (
        "skip",
        "exact",
        "greedy_local",
        "standard_qaoa",
        "qiskit_qaoa",
        "warm_start_qaoa",
        "hybrid_preconditioned",
    )


def available_error_models() -> tuple[str, ...]:
    """Return the supported error-model identifiers for quantum subroutines."""
    return ("ideal", "ankaa3", "ankaa3_hardware")


def _make_quantum_updater(
    *,
    depth: int,
    restarts: int,
    maxiter: int,
    use_warm_start: bool,
    use_library_qaoa: bool,
    error_model: str,
    error_model_specs_path: str,
    error_model_layout_seed: int,
    standard_optimizer_config: StandardQAOAOptimizerConfig,
    library_optimizer_config: LibraryQAOAOptimizerConfig,
    warm_start_optimizer_config: WarmStartQAOAOptimizerConfig,
) -> BlockUpdater:
    from .baselines import apply_quantum_block_update

    def updater(graph: nx.Graph, block: Sequence[int], assignment: Dict[int, int], seed: int) -> None:
        apply_quantum_block_update(
            graph,
            block,
            assignment,
            depth=depth,
            restarts=restarts,
            maxiter=maxiter,
            seed=seed,
            use_warm_start=use_warm_start,
            use_library_qaoa=use_library_qaoa,
            error_model=error_model,
            error_model_specs_path=error_model_specs_path,
            error_model_layout_seed=error_model_layout_seed,
            standard_optimizer_config=standard_optimizer_config,
            library_optimizer_config=library_optimizer_config,
            warm_start_optimizer_config=warm_start_optimizer_config,
        )

    return updater


def _make_exact_updater() -> BlockUpdater:
    from .baselines import apply_exact_block_update

    def updater(graph: nx.Graph, block: Sequence[int], assignment: Dict[int, int], seed: int) -> None:
        del seed
        apply_exact_block_update(graph, block, assignment)

    return updater


def _make_greedy_updater() -> BlockUpdater:
    from .baselines import apply_greedy_block_update

    def updater(graph: nx.Graph, block: Sequence[int], assignment: Dict[int, int], seed: int) -> None:
        apply_greedy_block_update(graph, block, assignment, seed=seed)

    return updater


def _make_hybrid_preconditioned_updater(
    *,
    depth: int,
    restarts: int,
    maxiter: int,
) -> BlockUpdater:
    del depth, restarts, maxiter

    def updater(graph: nx.Graph, block: Sequence[int], assignment: Dict[int, int], seed: int) -> None:
        del graph, block, assignment, seed
        raise RuntimeError(
            "hybrid_preconditioned is a whole-graph strategy. "
            "Use it for both region and boundary optimizers through solve_graph_modular()."
        )

    return updater


def get_block_updater(strategy: str, config: ModularSolveConfig) -> BlockUpdater | None:
    """Translate a strategy label into the callable used by block coordinate descent."""
    normalized = strategy.strip().lower()
    if normalized == "skip":
        return None
    if normalized == "exact":
        return _make_exact_updater()
    if normalized == "greedy_local":
        return _make_greedy_updater()
    if normalized == "standard_qaoa":
        return _make_quantum_updater(
            depth=config.depth,
            restarts=config.restarts,
            maxiter=config.maxiter,
            use_warm_start=False,
            use_library_qaoa=False,
            error_model=config.error_model,
            error_model_specs_path=config.error_model_specs_path,
            error_model_layout_seed=config.error_model_layout_seed,
            standard_optimizer_config=config.standard_qaoa_optimizer,
            library_optimizer_config=config.library_qaoa_optimizer,
            warm_start_optimizer_config=config.warm_start_qaoa_optimizer,
        )
    if normalized == "qiskit_qaoa":
        return _make_quantum_updater(
            depth=config.depth,
            restarts=config.restarts,
            maxiter=config.maxiter,
            use_warm_start=False,
            use_library_qaoa=True,
            error_model=config.error_model,
            error_model_specs_path=config.error_model_specs_path,
            error_model_layout_seed=config.error_model_layout_seed,
            standard_optimizer_config=config.standard_qaoa_optimizer,
            library_optimizer_config=config.library_qaoa_optimizer,
            warm_start_optimizer_config=config.warm_start_qaoa_optimizer,
        )
    if normalized == "warm_start_qaoa":
        return _make_quantum_updater(
            depth=config.depth,
            restarts=config.restarts,
            maxiter=config.maxiter,
            use_warm_start=True,
            use_library_qaoa=False,
            error_model=config.error_model,
            error_model_specs_path=config.error_model_specs_path,
            error_model_layout_seed=config.error_model_layout_seed,
            standard_optimizer_config=config.standard_qaoa_optimizer,
            library_optimizer_config=config.library_qaoa_optimizer,
            warm_start_optimizer_config=config.warm_start_qaoa_optimizer,
        )
    if normalized == "hybrid_preconditioned":
        return _make_hybrid_preconditioned_updater(
            depth=config.depth,
            restarts=config.restarts,
            maxiter=config.maxiter,
        )
    raise ValueError(f"Unknown optimization strategy: {strategy}")


def _initial_assignment_for_schedule(
    graph: nx.Graph,
    schedule: PartitionSchedule,
    config: ModularSolveConfig,
) -> Dict[int, int]:
    if schedule.layout is not None:

        from .core import assign_regions_from_coarse, solve_small_graph

        coarse_assignment = solve_small_graph(
            schedule.layout.coarse_graph,
            seed=config.seed,
            exact_threshold=config.exact_threshold,
        )
        return assign_regions_from_coarse(schedule.layout.regions, coarse_assignment)

    assignment = initial_assignment(graph, config.seed)
    return greedy_refine(graph, assignment, seed=config.seed)


def solve_graph_modular(
    graph: nx.Graph,
    name: str,
    config: ModularSolveConfig | None = None,
):
    """Solve a graph with the modular pipeline or defer to the whole-graph hybrid path."""
    from .core import SolveResult, chunked_exact_cut

    config = ModularSolveConfig() if config is None else config
    region_strategy = config.region_optimizer.strip().lower()
    boundary_strategy = config.boundary_optimizer.strip().lower()
    uses_preconditioned_solver = (
        region_strategy == "hybrid_preconditioned" or boundary_strategy == "hybrid_preconditioned"
    )
    if uses_preconditioned_solver:
        if region_strategy != "hybrid_preconditioned" or boundary_strategy != "hybrid_preconditioned":
            raise ValueError(
                "hybrid_preconditioned must be used for both region and boundary optimizers. "
                "It now builds a whole preconditioned graph before the classical solve."
            )
        from .preconditioning import solve_graph_quantum_preconditioned

        return solve_graph_quantum_preconditioned(graph, name=name, config=config)

    schedule = build_partition_schedule(
        graph,
        max_block_size=config.max_block_size,
        seed=config.seed,
        strategy=config.partition_strategy,
    )
    assignment = _initial_assignment_for_schedule(graph, schedule, config)
    record_checkpoint(weighted_cut_value(graph, assignment), label="initial")
    region_updater = get_block_updater(config.region_optimizer, config)
    boundary_updater = get_block_updater(config.boundary_optimizer, config)

    pipeline_result = run_block_coordinate_descent(
        graph,
        assignment,
        region_blocks=schedule.region_blocks,
        boundary_blocks=schedule.boundary_blocks,
        rounds=config.rounds,
        seed=config.seed,
        region_updater=region_updater,
        boundary_updater=boundary_updater,
        refine_assignment=lambda run_graph, current, refine_seed: greedy_refine(
            run_graph,
            current,
            seed=refine_seed,
        ),
        objective_value=weighted_cut_value,
        shuffle_region_blocks=schedule.layout is None,
        shuffle_boundary_blocks=False,
    )

    assignment = pipeline_result.assignment
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
            f"{config.region_optimizer}->{config.boundary_optimizer}"
        ),
    )
