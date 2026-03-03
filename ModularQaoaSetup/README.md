# QAOA Solver Stack

`ModularQaoaSetup/` contains the reusable solver, partitioning, noise-model, classical benchmarking, visualization, and resource-accounting code used in this repository.

## Layout

- `qaoa_solvers/`: QAOA circuits, optimizers, baselines, solver entrypoints, and preconditioning pipelines.
- `partitioning_methods/`: graph partitioning strategies and schedule builders.
- `error_modeling/`: device-noise hooks plus postprocessing mitigation and pulse-suppression layers.
- `assembly.py`: helpers for assembling the solver stack and running end-to-end solves.
- `resource_accounting.py`: the credit-based resource model used to compare quantum and classical work on the same basis.
- `benchmarking.py`: a convenience wrapper for tracked solver runs with timings and credit checkpoints.
- `benchmarking_classical.py`: classical seed sweeps, BM baselines, and polishing utilities for strong reference solutions.
- `visualization.py`: graph, partition, and coarse-graph visualization helpers.
- `notebooks/`: minimal usage walkthrough.

## Import surface

Import from `ModularQaoaSetup` when working from the repository root:

```python
from ModularQaoaSetup import (
    ResourceTracker,
    StandardQAOAOptimizerConfig,
    WarmStartQAOAOptimizerConfig,
    LibraryQAOAOptimizerConfig,
    assemble_portable_solver,
    load_weighted_graph,
    run_portable_solver,
    tracking_resources,
)
```

Classical benchmarking utilities:

```python
from ModularQaoaSetup.classical_benchmarks import (
    sweep_seeds_best,
    best_bm_over_seeds_then_polish,
)
```

Visualization helpers:

```python
from ModularQaoaSetup.visualization import (
    draw_graph,
    plot_boundary_nodes,
    plot_regions,
    plot_coarse_graph,
)
```

## Optimizer configs

The optimizer configs expose:

- simplex method name
- convergence tolerances (`xatol`, `fatol`)
- adaptive simplex toggle
- first-pass initialization angles
- randomized restart sampling ranges

## Warm Start & Preconditioning Modes

`ModularSolveConfig` additionally supports:

```python
warm_start_assignment: dict[int, int] | None
preconditioning_mode: Literal["bcd", "global_bm_step"]
```

- `warm_start_assignment` allows injecting a user-provided ±1 assignment instead of random initialization.
- `preconditioning_mode="bcd"` (default) runs the standard block-coordinate descent pipeline.
- `preconditioning_mode="global_bm_step"` runs a single global BM refinement on the QAOA-preconditioned working graph before optional polishing.

This enables controlled benchmarking between:

- Pure classical baselines  
- QAOA-preconditioned workflows  
- Hybrid warm-start strategies  

## Minimal example

```python
from pathlib import Path

from ModularQaoaSetup import assemble_portable_solver, load_weighted_graph, run_portable_solver

repo_root = Path.cwd()
data_dir = repo_root / "data"
graph_path = max(data_dir.glob("*.parquet"), key=lambda path: path.stat().st_size)

graph = load_weighted_graph(graph_path)
stack = assemble_portable_solver()
result = run_portable_solver(graph, name=graph_path.name, stack=stack)

print(result.cut_value)
```

## Classical Baseline Example

```python
from ModularQaoaSetup.classical_benchmarks import best_bm_over_seeds_then_polish

result = best_bm_over_seeds_then_polish(graph, cfg, seeds=range(2000))

print("Best seed:", result["best_seed"])
print("BM best:", result["bm_best_score"])
print("After polish:", result["polished_score"])
```

This utility provides a strong classical reference solution before evaluating QAOA improvements.