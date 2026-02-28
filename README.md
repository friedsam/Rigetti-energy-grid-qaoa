# Energy Grid Optimization (MPES / Weighted MaxCut)

Goal: Solve large weighted MaxCut instances under quantum hardware limits via
divide-and-conquer + stitching + (optional) QAOA on subgraphs.

## Minimal runnable pipeline (v0)
1) Load graph (edgelist -> NetworkX)
2) Partition into components of size <= K
3) Solve each component classically (greedy + local refinement)
4) Stitch via component flip-alignment
5) Score full graph

Then replace step (3) with QAOA-on-subgraphs (Rigetti simulator/QPU).

## Repo layout
- src/graph: load + scoring
- src/partition: graph splitting
- src/stitch: stitch + boundary refinement
- src/solvers/classical: baselines
- src/solvers/quantum: QAOA backend(s)
- notebooks: exploration notebooks calling src/*
- data: input edgelists (NOT committed if large)
- results: saved scores/bitstrings