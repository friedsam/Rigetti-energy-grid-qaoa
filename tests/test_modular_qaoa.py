from __future__ import annotations

from typing import Dict

import networkx as nx
import pytest

import ModularQaoaSetup as mqs
import ModularQaoaSetup.qaoa_solvers.preconditioning as preconditioning


def _build_small_graph() -> nx.Graph:
    graph = nx.Graph()
    graph.add_weighted_edges_from(
        (
            (0, 1, 1.0),
            (1, 2, 2.0),
            (2, 3, 1.5),
            (3, 0, 2.5),
            (0, 2, 0.75),
        )
    )
    return graph


def _alternating_assignment(graph: nx.Graph) -> Dict[int, int]:
    return {
        int(node): (1 if index % 2 == 0 else -1)
        for index, node in enumerate(sorted(int(node) for node in graph.nodes()))
    }


def test_default_modular_solve_config_uses_matching_preconditioned_optimizers() -> None:
    config = mqs.ModularSolveConfig()

    assert config.region_optimizer == "hybrid_preconditioned"
    assert config.boundary_optimizer == "hybrid_preconditioned"
    assert mqs.assemble_portable_solver().config == config


def test_invalid_half_preconditioned_config_is_rejected() -> None:
    with pytest.raises(ValueError, match="hybrid_preconditioned must be used for both"):
        mqs.ModularSolveConfig(region_optimizer="exact", boundary_optimizer="hybrid_preconditioned")


def test_default_solver_routes_to_preconditioned_path(monkeypatch: pytest.MonkeyPatch) -> None:
    graph = _build_small_graph()
    assignment = _alternating_assignment(graph)
    sentinel = mqs.SolveResult(
        name="patched",
        graph=graph,
        assignment=assignment,
        blocks=[tuple(sorted(graph.nodes()))],
        rounds_run=1,
        cut_value=3.25,
        ising_energy=-3.25,
        exact_cut_value=None,
        strategy="patched",
    )

    calls: list[tuple[str, str, str]] = []

    def fake_solver(run_graph: nx.Graph, name: str, config: mqs.ModularSolveConfig) -> mqs.SolveResult:
        assert run_graph is graph
        calls.append((name, config.region_optimizer, config.boundary_optimizer))
        return sentinel

    monkeypatch.setattr(preconditioning, "solve_graph_quantum_preconditioned", fake_solver)

    result = mqs.solve_graph_modular(graph, name="default-config")

    assert result is sentinel
    assert calls == [("default-config", "hybrid_preconditioned", "hybrid_preconditioned")]


def test_run_portable_solver_exact_smoke() -> None:
    graph = _build_small_graph()
    config = mqs.ModularSolveConfig(
        region_optimizer="exact",
        boundary_optimizer="exact",
        rounds=1,
        max_block_size=4,
        seed=5,
    )

    result = mqs.run_portable_solver(
        graph,
        name="exact-smoke",
        stack=mqs.assemble_portable_solver(config=config),
    )
    payload = result.to_dict()

    assert payload["name"] == "exact-smoke"
    assert payload["nodes"] == graph.number_of_nodes()
    assert payload["edges"] == graph.number_of_edges()
    assert payload["weighted_cut"] >= 0.0
    assert payload["strategy"] == "multilevel:exact->exact"
