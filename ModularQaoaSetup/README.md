# QAOA Solver Stack

`ModularQaoaSetup/` contains the reusable solver, partitioning, noise-model, and resource-accounting code used in this repository.

## Layout

- `qaoa_solvers/`: QAOA circuits, optimizers, baselines, and solver entrypoints.
- `partitioning_methods/`: graph partitioning strategies and schedule builders.
- `error_modeling/`: device-noise hooks plus postprocessing mitigation and pulse-suppression layers.
- `assembly.py`: helpers for assembling the solver stack and running end-to-end solves.
- `resource_accounting.py`: the credit-based resource model used to compare quantum and classical work on the same basis.
- `benchmarking.py`: a convenience wrapper for tracked solver runs with timings and credit checkpoints.
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

## Optimizer configs

The optimizer configs expose:

- simplex method name
- convergence tolerances (`xatol`, `fatol`)
- adaptive simplex toggle
- first-pass initialization angles
- randomized restart sampling ranges

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
