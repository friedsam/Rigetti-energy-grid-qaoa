from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class WarmStartQAOAOptimizerConfig:
    method: str = "Nelder-Mead"
    xatol: float = 1e-3
    fatol: float = 1e-3
    adaptive: bool = True
    initial_gamma: float = -0.05
    initial_beta: float = math.pi / 4.0
    random_gamma_min: float = -0.35
    random_gamma_max: float = 0.35
    random_beta_min: float = 0.05
    random_beta_max: float = math.pi / 2.0


@dataclass(frozen=True)
class StandardQAOAOptimizerConfig:
    method: str = "Nelder-Mead"
    xatol: float = 1e-3
    fatol: float = 1e-3
    adaptive: bool = True
    initial_gamma: float = -0.10
    initial_beta: float = math.pi / 8.0
    random_gamma_min: float = -0.35
    random_gamma_max: float = 0.35
    random_beta_min: float = 0.05
    random_beta_max: float = math.pi / 2.0


@dataclass(frozen=True)
class LibraryQAOAOptimizerConfig:
    method: str = "Nelder-Mead"
    xatol: float = 1e-3
    fatol: float = 1e-3
    adaptive: bool = True
    initial_cost_angle: float = math.pi / 8.0
    initial_mixer_angle: float = -0.10
    random_cost_angle_min: float = 0.05
    random_cost_angle_max: float = math.pi / 2.0
    random_mixer_angle_min: float = -0.35
    random_mixer_angle_max: float = 0.35
