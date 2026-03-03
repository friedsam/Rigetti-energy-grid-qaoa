from __future__ import annotations

from dataclasses import replace
from typing import Callable, Dict, Iterable, Optional, Tuple, Any
import random
import networkx as nx

from ModularQaoaSetup.graph_ops import (
    burer_monteiro_refine,
    greedy_refine,
    weighted_cut_value,
)

Assignment = Dict[int, int]  # {node: +/-1}
InitFn = Callable[[nx.Graph, int], Assignment]
SolverFn = Callable[[nx.Graph, Assignment, int], Assignment]
PolishFn = Callable[[nx.Graph, Assignment, Any], Tuple[float, Assignment]]


def random_pm1_assignment(G: nx.Graph, seed: int) -> Assignment:
    rng = random.Random(seed)
    return {int(u): (1 if rng.random() < 0.5 else -1) for u in G.nodes()}


def score_assignment(G: nx.Graph, assignment: Assignment) -> float:
    return float(weighted_cut_value(G, assignment))


def run_solver_once(
    G: nx.Graph,
    solver: SolverFn,
    seed: int,
    init: Optional[Assignment] = None,
    init_fn: InitFn = random_pm1_assignment,
) -> Tuple[float, Assignment]:
    if init is None:
        init = init_fn(G, seed)
    a = solver(G, init, seed)
    return score_assignment(G, a), a


def sweep_seeds_best(
    G: nx.Graph,
    solver: SolverFn,
    seeds: Iterable[int],
    init_fn: InitFn = random_pm1_assignment,
    *,
    verbose: bool = False,
) -> Tuple[float, int, Assignment]:
    best_score = float("-inf")
    best_seed: Optional[int] = None
    best_assignment: Optional[Assignment] = None

    for s in seeds:
        a0 = init_fn(G, s)
        score, a = run_solver_once(G, solver, seed=s, init=a0, init_fn=init_fn)

        if verbose:
            print(s, score)

        if score > best_score:
            best_score = score
            best_seed = int(s)
            best_assignment = a.copy()

    if best_seed is None or best_assignment is None:
        raise ValueError("No seeds provided or solver produced no results.")

    return best_score, best_seed, best_assignment


# ----------------------------
# Built-in solver: BM fast baseline
# ----------------------------
def solver_bm_fast(G: nx.Graph, init: Assignment, seed: int) -> Assignment:
    # Same behavior as your "fast loop": per-seed random init + burer_monteiro_refine on full graph
    return burer_monteiro_refine(G, init, seed=seed)


# ----------------------------
# Built-in polish: matches your preconditioning-style polish
# Requires cfg to have bm_rank, bm_steps, bm_learning_rate, bm_rounding_trials, seed
# ----------------------------
def polish_like_preconditioning(G: nx.Graph, assignment: Assignment, cfg: Any) -> Tuple[float, Assignment]:
    a = assignment.copy()
    a = burer_monteiro_refine(
        G, a,
        rank=cfg.bm_rank,
        steps=max(12, cfg.bm_steps),
        learning_rate=cfg.bm_learning_rate,
        rounding_trials=max(12, cfg.bm_rounding_trials),
        seed=int(cfg.seed) + 9000,
    )
    a = greedy_refine(G, a, seed=int(cfg.seed) + 10000)
    return score_assignment(G, a), a


def best_bm_over_seeds_then_polish(
    G: nx.Graph,
    cfg: Any,
    seeds: Iterable[int],
    *,
    verbose: bool = False,
) -> dict:
    bm_best_score, best_seed, bm_best_assignment = sweep_seeds_best(
        G, solver=solver_bm_fast, seeds=seeds, verbose=verbose
    )
    cfg_best = replace(cfg, seed=best_seed) if hasattr(cfg, "__dataclass_fields__") else cfg
    polished_score, polished_assignment = polish_like_preconditioning(G, bm_best_assignment, cfg_best)

    return {
        "best_seed": best_seed,
        "bm_best_score": float(bm_best_score),
        "bm_best_assignment": bm_best_assignment,
        "polished_score": float(polished_score),
        "polished_assignment": polished_assignment,
    }