from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PostprocessingErrorMitigationLayer:
    enabled: bool = False
    name: str = "postprocessing_error_mitigation_placeholder"
    metadata: dict[str, Any] = field(default_factory=dict)

    def apply(self, solve_result: Any) -> Any:
        return solve_result

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "status": "placeholder",
            "metadata": dict(self.metadata),
        }


def build_postprocessing_error_mitigation_layer(
    *,
    enabled: bool = False,
    name: str = "postprocessing_error_mitigation_placeholder",
    metadata: dict[str, Any] | None = None,
) -> PostprocessingErrorMitigationLayer:
    return PostprocessingErrorMitigationLayer(
        enabled=enabled,
        name=name,
        metadata={} if metadata is None else dict(metadata),
    )

