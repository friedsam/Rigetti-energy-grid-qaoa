from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx

from .error_modeling import (
    DynamicPulseErrorSuppressionLayer,
    PostprocessingErrorMitigationLayer,
    build_dynamic_pulse_error_suppression_layer,
    build_postprocessing_error_mitigation_layer,
)
from .qaoa_solvers.core import SolveResult
from .qaoa_solvers.modular import ModularSolveConfig, solve_graph_modular


@dataclass(frozen=True)
class PortableSolverStack:
    config: ModularSolveConfig
    postprocessing_mitigation: PostprocessingErrorMitigationLayer
    dynamic_pulse_suppression: DynamicPulseErrorSuppressionLayer

    def describe(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "postprocessing_mitigation": self.postprocessing_mitigation.describe(),
            "dynamic_pulse_suppression": self.dynamic_pulse_suppression.describe(),
        }


def assemble_portable_solver(
    *,
    config: ModularSolveConfig | None = None,
    postprocessing_mitigation: PostprocessingErrorMitigationLayer | None = None,
    dynamic_pulse_suppression: DynamicPulseErrorSuppressionLayer | None = None,
) -> PortableSolverStack:
    return PortableSolverStack(
        config=ModularSolveConfig() if config is None else config,
        postprocessing_mitigation=(
            build_postprocessing_error_mitigation_layer()
            if postprocessing_mitigation is None
            else postprocessing_mitigation
        ),
        dynamic_pulse_suppression=(
            build_dynamic_pulse_error_suppression_layer()
            if dynamic_pulse_suppression is None
            else dynamic_pulse_suppression
        ),
    )


def run_portable_solver(
    graph: nx.Graph,
    *,
    name: str,
    stack: PortableSolverStack | None = None,
) -> SolveResult:
    active_stack = assemble_portable_solver() if stack is None else stack
    result = solve_graph_modular(graph, name=name, config=active_stack.config)
    return active_stack.postprocessing_mitigation.apply(result)


__all__ = (
    "PortableSolverStack",
    "assemble_portable_solver",
    "run_portable_solver",
)
