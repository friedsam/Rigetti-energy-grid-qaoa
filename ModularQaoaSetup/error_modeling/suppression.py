from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DynamicPulseErrorSuppressionLayer:
    enabled: bool = False
    name: str = "dynamic_pulse_error_suppression_placeholder"
    metadata: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "status": "placeholder",
            "metadata": dict(self.metadata),
        }


def build_dynamic_pulse_error_suppression_layer(
    *,
    enabled: bool = False,
    name: str = "dynamic_pulse_error_suppression_placeholder",
    metadata: dict[str, Any] | None = None,
) -> DynamicPulseErrorSuppressionLayer:
    return DynamicPulseErrorSuppressionLayer(
        enabled=enabled,
        name=name,
        metadata={} if metadata is None else dict(metadata),
    )

