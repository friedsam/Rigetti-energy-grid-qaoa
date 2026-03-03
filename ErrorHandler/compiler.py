"""Compile Qiskit circuits into Quil with hardware-aware mitigation utilities.

The compiler module is the core of the ``ErrorHandler`` package. It loads device
calibration data, chooses an initial placement, routes and schedules operations,
emits optional pulse-calibration overlays, and provides helpers for zero-noise
extrapolation, readout mitigation, and closed-loop pulse tuning.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence

import networkx as nx
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, transpile


EPS = 1e-9
DEFAULT_1Q_DURATION_NS = 50.0
DEFAULT_2Q_DURATION_NS = 200.0
DEFAULT_ANHARMONICITY_HZ = 2.0e8


@dataclass(frozen=True)
class PipelineSwitches:
    """Feature flags for the major stages of the compilation and mitigation flow."""
    normalize_circuit: bool = True
    placement: bool = True
    routing: bool = True
    pulse_calibrations: bool = True
    robust_overlays: bool = True
    dynamical_decoupling: bool = True
    zne: bool = True
    readout_mitigation: bool = True


@dataclass(frozen=True)
class CompilerConfig:
    """User-facing configuration for the hardware-aware Quil compilation pipeline."""
    calibration_path: str | Path = Path("data") / "Ankaa-3_device_specs.csv"
    optimization_level: int = 3
    placement_trials: int = 8
    enable_pulse_calibrations: bool = True
    enable_dd: bool = True
    dd_idle_cycles: int = 4
    dd_risk_threshold: float = 0.0
    synchronize_dd: bool = True
    switches: PipelineSwitches = field(default_factory=PipelineSwitches)


@dataclass(frozen=True)
class LogicalOperation:
    """A logical gate extracted from the input Qiskit circuit."""
    name: str
    qubits: tuple[int, ...]
    params: tuple[float, ...] = ()
    clbits: tuple[int, ...] = ()

    @property
    def arity(self) -> int:
        return len(self.qubits)


@dataclass(frozen=True)
class PhysicalOperation:
    """A placed and optionally routed gate expressed on physical qubits."""
    name: str
    qubits: tuple[int, ...]
    params: tuple[float, ...] = ()
    clbits: tuple[int, ...] = ()
    source: str = "logical"

    @property
    def arity(self) -> int:
        return len(self.qubits)


@dataclass(frozen=True)
class CalibrationData:
    """Normalized hardware metrics derived from the calibration CSV."""
    path: Path
    topology: nx.Graph
    node_cost: dict[int, float]
    edge_cost: dict[tuple[int, int], float]
    qubit_metrics: dict[int, dict[str, float]]
    pair_metrics: dict[tuple[int, int], dict[str, float]]
    default_1q_duration_ns: float = DEFAULT_1Q_DURATION_NS
    default_2q_duration_ns: float = DEFAULT_2Q_DURATION_NS

    def crosstalk_risk(self, qubit: int) -> float:
        total = 0.0
        for neighbor in self.topology.neighbors(qubit):
            total += self.edge_cost.get((int(qubit), int(neighbor)), 1.0)
        return total


@dataclass(frozen=True)
class CompilationResult:
    """The compiled Quil program plus intermediate scheduling metadata."""
    circuit: QuantumCircuit
    calibration: CalibrationData
    logical_to_physical: dict[int, int]
    final_logical_to_physical: dict[int, int]
    operations: tuple[PhysicalOperation, ...]
    moments: tuple[tuple[PhysicalOperation, ...], ...]
    pulse_calibration_quil: str
    quil: str
    score: float
    pipeline_switches: PipelineSwitches = field(default_factory=PipelineSwitches)
    overlay_recommendations: tuple[str, ...] = ()

    def to_program(self) -> Any:
        """Materialize the generated Quil source as a ``pyquil.Program``."""
        try:
            from pyquil import Program
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "pyquil is not installed in this environment. Install pyquil to build a Program object "
                "from the generated Quil source."
            ) from exc
        return Program(self.quil)

    def compile_with_backend(self, qc: Any) -> Any:
        """Compile the generated Quil program with a live pyQuil backend handle."""
        compiled = qc.compile(self.to_program())
        return getattr(compiled, "program", compiled)


@dataclass(frozen=True)
class QCSExecutionResult:
    """Results returned from executing folded Quil variants with mitigation enabled."""
    raw_probabilities: dict[str, float]
    readout_mitigated_probabilities: dict[str, float]
    folded_expectations: dict[int, float]
    mitigated_expectation: float
    shots_per_scale: int


@dataclass(frozen=True)
class PulseControlParameters:
    """Parameterized controls for a single gate family in pulse optimization loops."""
    gate: str
    qubits: tuple[int, ...]
    amplitude_scale: float = 1.0
    duration_scale: float = 1.0
    drag_scale: float = 1.0
    detuning_hz: float = 0.0
    phase_shift: float = math.pi
    drive_segments: tuple[float, ...] = ()
    quadrature_segments: tuple[float, ...] = ()
    frequency_segments_hz: tuple[float, ...] = ()
    echo_scale: float = 1.0


@dataclass(frozen=True)
class ClosedLoopOptimizationConfig:
    """Hyperparameters for the SPSA-style pulse tuning routine."""
    calibration_path: str | Path = Path("data") / "Ankaa-3_device_specs.csv"
    shots_per_eval: int = 64
    max_evaluations: int = 7
    amplitude_step: float = 0.08
    duration_step: float = 0.10
    drag_step: float = 0.20
    detuning_step_hz: float = 5.0e6
    phase_step: float = 0.10
    control_segments: int = 5
    spsa_learning_rate: float = 0.18
    spsa_perturbation: float = 0.08
    spsa_decay: float = 0.5
    segment_min: float = 0.2
    segment_max: float = 1.8
    quadrature_limit: float = 0.8
    frequency_limit_hz: float = 2.0e7
    gradient_clip: float = 3.0
    spsa_seed: int = 7


@dataclass(frozen=True)
class ClosedLoopOptimizationResult:
    """Summary of the pulse search, including the best parameters and emitted Quil."""
    baseline_parameters: PulseControlParameters
    best_parameters: PulseControlParameters
    baseline_score: float
    best_score: float
    evaluations: int
    calibration_quil: str


def load_calibration(
    csv_path: str | Path,
    *,
    quantum_processor: Any | None = None,
) -> CalibrationData:
    """Load a calibration CSV and normalize it into placement and timing metadata."""
    path = Path(csv_path)
    frame = pd.read_csv(path)
    frame = frame.rename(columns=lambda name: _normalize_column_name(str(name)))

    topology = nx.Graph()
    node_cost: dict[int, float] = {}
    edge_cost: dict[tuple[int, int], float] = {}
    qubit_metrics: dict[int, dict[str, float]] = {}
    pair_metrics: dict[tuple[int, int], dict[str, float]] = {}
    one_qubit_durations_ns: list[float] = []
    two_qubit_durations_ns: list[float] = []

    qubit_rows = frame[frame.get("qubit").notna()] if "qubit" in frame.columns else frame.iloc[0:0]
    for row in qubit_rows.to_dict("records"):
        qubit = int(float(row["qubit"]))
        t1 = _coerce_metric(row, "t1", "t1_s", default=20.0)
        t2 = _coerce_metric(row, "t2", "t2_s", default=20.0)
        readout = _coerce_metric(row, "fro", "avg_fro", default=0.95)
        leakage = _optional_metric(row, "p1") or 0.0
        frequency_ghz = _optional_metric(row, "freq_ghz") or 0.0
        anharmonicity_hz = (
            (_optional_metric(row, "anharmonicity_hz") or 0.0)
            or (_optional_metric(row, "anharmonicity_mhz") or 0.0) * 1e6
            or DEFAULT_ANHARMONICITY_HZ
        )
        one_qubit_duration = _optional_metric(row, "duration_ns", "gate1q_duration_ns", "gate_duration_ns")
        one_qubit_fidelity = _coerce_metric(
            row,
            "f1q_sim_rb",
            "f1qsim_rb",
            "f1qrb",
            "avg_f1qrb",
            default=0.99,
        )
        cost = (
            1.0 / max(t1, EPS)
            + 1.0 / max(t2, EPS)
            + (1.0 - readout)
            + 0.5 * (1.0 - one_qubit_fidelity)
            + 0.25 * max(leakage, 0.0)
        )
        if one_qubit_duration is not None:
            one_qubit_durations_ns.append(float(one_qubit_duration))

        topology.add_node(qubit)
        node_cost[qubit] = float(cost)
        qubit_metrics[qubit] = {
            "t1": float(t1),
            "t2": float(t2),
            "readout_fidelity": float(readout),
            "one_qubit_fidelity": float(one_qubit_fidelity),
            "p1": float(leakage),
            "freq_ghz": float(frequency_ghz),
            "anharmonicity_hz": float(anharmonicity_hz),
            "duration_ns": float(one_qubit_duration or DEFAULT_1Q_DURATION_NS),
        }

    pair_rows = frame[frame.get("pair").notna()] if "pair" in frame.columns else frame.iloc[0:0]
    for row in pair_rows.to_dict("records"):
        pair = _parse_pair(str(row["pair"]))
        if pair is None:
            continue
        q1, q2 = pair
        topology.add_node(q1)
        topology.add_node(q2)
        fidelity = _coerce_metric(row, "fiswap", default=0.9)
        zz_mhz = _optional_metric(row, "zz_mhz", "zz") or 0.0
        two_qubit_duration = _optional_metric(row, "duration_ns", "gate2q_duration_ns", "gate_duration_ns")
        pair_score = (1.0 - fidelity) + 0.01 * abs(zz_mhz) + 0.25 * (
            node_cost.get(q1, 1.0 / 20.0) + node_cost.get(q2, 1.0 / 20.0)
        )
        if two_qubit_duration is not None:
            two_qubit_durations_ns.append(float(two_qubit_duration))

        topology.add_edge(q1, q2, cost=float(pair_score), fidelity=float(fidelity))
        edge_cost[(q1, q2)] = float(pair_score)
        edge_cost[(q2, q1)] = float(pair_score)
        pair_metrics[(q1, q2)] = {
            "fidelity": float(fidelity),
            "zz_mhz": float(zz_mhz),
            "duration_ns": float(two_qubit_duration or DEFAULT_2Q_DURATION_NS),
        }
        pair_metrics[(q2, q1)] = dict(pair_metrics[(q1, q2)])

    calibration = CalibrationData(
        path=path,
        topology=topology,
        node_cost=node_cost,
        edge_cost=edge_cost,
        qubit_metrics=qubit_metrics,
        pair_metrics=pair_metrics,
        default_1q_duration_ns=(
            float(sum(one_qubit_durations_ns) / len(one_qubit_durations_ns))
            if one_qubit_durations_ns
            else DEFAULT_1Q_DURATION_NS
        ),
        default_2q_duration_ns=(
            float(sum(two_qubit_durations_ns) / len(two_qubit_durations_ns))
            if two_qubit_durations_ns
            else DEFAULT_2Q_DURATION_NS
        ),
    )
    if quantum_processor is not None:
        calibration = _overlay_quantum_processor_topology(calibration, quantum_processor)
    return calibration


def _overlay_quantum_processor_topology(
    calibration: CalibrationData,
    quantum_processor: Any,
) -> CalibrationData:
    if not hasattr(quantum_processor, "qubit_topology"):
        return calibration

    live_topology = quantum_processor.qubit_topology()
    if live_topology is None:
        return calibration

    topology = nx.Graph()
    topology.add_nodes_from(int(node) for node in live_topology.nodes)
    topology.add_edges_from((int(left), int(right)) for left, right in live_topology.edges)

    existing_node_costs = list(calibration.node_cost.values()) or [1.0]
    default_node_cost = float(sum(existing_node_costs) / len(existing_node_costs))
    existing_edge_costs = list(calibration.edge_cost.values()) or [1.0]
    default_edge_cost = float(sum(existing_edge_costs) / len(existing_edge_costs))

    node_cost: dict[int, float] = {}
    qubit_metrics: dict[int, dict[str, float]] = {}
    for node in topology.nodes:
        node = int(node)
        node_cost[node] = float(calibration.node_cost.get(node, default_node_cost))
        qubit_metrics[node] = dict(
            calibration.qubit_metrics.get(
                node,
                {
                    "t1": 20.0,
                    "t2": 20.0,
                    "readout_fidelity": 0.95,
                    "one_qubit_fidelity": 0.99,
                    "p1": 0.0,
                    "freq_ghz": 5.0,
                    "anharmonicity_hz": DEFAULT_ANHARMONICITY_HZ,
                    "duration_ns": calibration.default_1q_duration_ns,
                },
            )
        )

    edge_cost: dict[tuple[int, int], float] = {}
    pair_metrics: dict[tuple[int, int], dict[str, float]] = {}
    for left, right in topology.edges:
        left = int(left)
        right = int(right)
        pair = (left, right)
        reverse = (right, left)
        cost = float(calibration.edge_cost.get(pair, calibration.edge_cost.get(reverse, default_edge_cost)))
        metrics = dict(
            calibration.pair_metrics.get(
                pair,
                calibration.pair_metrics.get(
                    reverse,
                    {
                        "fidelity": 1.0 - min(0.5, default_edge_cost),
                        "zz_mhz": 0.0,
                        "duration_ns": calibration.default_2q_duration_ns,
                    },
                ),
            )
        )
        edge_cost[pair] = cost
        edge_cost[reverse] = cost
        pair_metrics[pair] = dict(metrics)
        pair_metrics[reverse] = dict(metrics)
        topology[left][right]["cost"] = cost
        topology[left][right]["fidelity"] = float(metrics.get("fidelity", 1.0))

    return CalibrationData(
        path=calibration.path,
        topology=topology,
        node_cost=node_cost,
        edge_cost=edge_cost,
        qubit_metrics=qubit_metrics,
        pair_metrics=pair_metrics,
        default_1q_duration_ns=calibration.default_1q_duration_ns,
        default_2q_duration_ns=calibration.default_2q_duration_ns,
    )


def compile_qiskit_to_quil(
    circuit: QuantumCircuit,
    *,
    config: CompilerConfig | None = None,
    initial_mapping: Mapping[int, int] | None = None,
    qc: Any | None = None,
) -> CompilationResult:
    """Compile a Qiskit circuit into Quil using the configured placement and mitigation stack."""
    config = CompilerConfig() if config is None else config
    switches = _resolve_pipeline_switches(config)
    calibration = load_calibration(
        config.calibration_path,
        quantum_processor=(getattr(qc, "quantum_processor", None) if qc is not None else None),
    )
    normalized = (
        _normalize_circuit(circuit, optimization_level=config.optimization_level)
        if switches.normalize_circuit
        else circuit.copy()
    )
    logical_ops = _extract_logical_operations(normalized)

    if initial_mapping is None:
        if switches.placement:
            candidate_mappings = _candidate_initial_mappings(
                normalized.num_qubits,
                logical_ops,
                calibration,
                trials=max(1, config.placement_trials),
            )
        else:
            candidate_mappings = [_default_identity_mapping(normalized.num_qubits, calibration)]
    else:
        candidate_mappings = [
            _validated_initial_mapping(initial_mapping, normalized.num_qubits, calibration)
        ]

    best_payload: tuple[
        dict[int, int],
        dict[int, int],
        list[PhysicalOperation],
        list[list[PhysicalOperation]],
        float,
    ] | None = None

    for candidate in candidate_mappings:
        routed_ops, final_mapping = _route_logical_operations(
            logical_ops,
            candidate,
            calibration,
            enable_routing=switches.routing,
        )
        moments = _schedule_operations(routed_ops, calibration)
        if switches.dynamical_decoupling:
            _insert_xy4_dd(
                moments,
                calibration,
                idle_cycles=max(4, config.dd_idle_cycles),
                risk_threshold=max(0.0, config.dd_risk_threshold),
                synchronize_pairs=config.synchronize_dd,
            )
        score = _score_physical_schedule(moments, calibration)
        if best_payload is None or score < best_payload[4]:
            best_payload = (dict(candidate), final_mapping, routed_ops, moments, score)

    if best_payload is None:
        raise RuntimeError("Compilation did not produce any candidate placement.")

    logical_to_physical, final_mapping, routed_ops, moments, score = best_payload
    flattened = tuple(op for moment in moments for op in moment)
    body_quil = _emit_quil(moments, normalized.num_clbits)
    pulse_calibration_quil = (
        _emit_pulse_calibrations(
            flattened,
            calibration,
            enable_robust_overlays=switches.robust_overlays,
        )
        if switches.pulse_calibrations
        else ""
    )
    quil = _compose_full_quil(pulse_calibration_quil, body_quil)
    overlays = _recommend_calibration_overlays(
        flattened,
        calibration,
        enable_robust_overlays=switches.robust_overlays,
    )
    return CompilationResult(
        circuit=normalized,
        calibration=calibration,
        logical_to_physical=dict(logical_to_physical),
        final_logical_to_physical=dict(final_mapping),
        operations=flattened,
        moments=tuple(tuple(moment) for moment in moments),
        pulse_calibration_quil=pulse_calibration_quil,
        quil=quil,
        score=float(score),
        pipeline_switches=switches,
        overlay_recommendations=overlays,
    )


def richardson_extrapolate(observations: Mapping[int, float]) -> float:
    """Estimate the zero-noise value from observations at odd folding scale factors."""
    if not observations:
        raise ValueError("At least one observation is required for Richardson extrapolation.")

    scales = np.array(sorted(int(scale) for scale in observations), dtype=float)
    if np.any(scales <= 0) or len(set(int(scale) for scale in scales)) != len(scales):
        raise ValueError("Scale factors must be distinct positive integers.")

    values = np.array([float(observations[int(scale)]) for scale in scales], dtype=float)
    order = len(scales) - 1
    system = np.vstack([scales**power for power in range(order + 1)])
    rhs = np.zeros(order + 1, dtype=float)
    rhs[0] = 1.0
    coeffs = np.linalg.solve(system, rhs)
    return float(coeffs @ values)


def mitigate_diagonal_readout(
    probabilities: Mapping[str, float],
    calibration: CalibrationData,
    logical_to_physical: Mapping[int, int],
) -> dict[str, float]:
    """Invert the factored single-qubit readout model stored in ``CalibrationData``."""
    if not probabilities:
        return {}

    bitstrings = sorted(probabilities)
    bit_length = len(bitstrings[0])
    if any(len(bitstring) != bit_length for bitstring in bitstrings):
        raise ValueError("All bitstrings must have the same length.")
    if set(logical_to_physical) != set(range(bit_length)):
        raise ValueError("logical_to_physical must cover the exact logical qubits in the observed bitstrings.")

    ordered = [format(index, f"0{bit_length}b") for index in range(1 << bit_length)]
    observed = np.array([float(probabilities.get(bitstring, 0.0)) for bitstring in ordered], dtype=float)

    response = np.array([[1.0]], dtype=float)
    for logical in range(bit_length):
        physical = int(logical_to_physical[logical])
        metrics = calibration.qubit_metrics.get(physical, {})
        fidelity = float(metrics.get("readout_fidelity", 0.95))
        flip_10 = max(0.0, min(0.5 - EPS, float(metrics.get("readout_p10", 1.0 - fidelity))))
        flip_01 = max(0.0, min(0.5 - EPS, float(metrics.get("readout_p01", 1.0 - fidelity))))
        local = np.array(
            [
                [1.0 - flip_10, flip_01],
                [flip_10, 1.0 - flip_01],
            ],
            dtype=float,
        )
        response = np.kron(local, response)

    mitigated = np.linalg.solve(response, observed)
    mitigated = np.clip(mitigated, 0.0, None)
    total = float(np.sum(mitigated))
    if total > EPS:
        mitigated /= total
    return {
        bitstring: float(mitigated[index])
        for index, bitstring in enumerate(ordered)
        if mitigated[index] > EPS
    }


def build_local_readout_calibration_programs(qubits: Sequence[int]) -> dict[str, str]:
    """Build simple Quil programs that prepare all-zeros and all-ones readout states."""
    ordered_qubits = tuple(int(qubit) for qubit in qubits)
    if not ordered_qubits:
        raise ValueError("At least one qubit is required to build a readout calibration program.")

    def _build_program(*, prepare_one: bool) -> str:
        lines = [f"DECLARE ro BIT[{len(ordered_qubits)}]"]
        if prepare_one:
            lines.extend(f"X {qubit}" for qubit in ordered_qubits)
        lines.extend(
            f"MEASURE {qubit} ro[{index}]"
            for index, qubit in enumerate(ordered_qubits)
        )
        return "\n".join(lines)

    return {
        "zero": _build_program(prepare_one=False),
        "one": _build_program(prepare_one=True),
    }


def calibrate_local_readout_with_backend(
    qc: Any,
    calibration: CalibrationData,
    *,
    qubits: Sequence[int] | None = None,
    shots: int = 256,
) -> CalibrationData:
    """Refresh per-qubit readout flip probabilities by executing calibration programs."""
    if qc is None:
        raise ValueError("A backend handle is required to run readout calibration.")

    ordered_qubits = (
        tuple(int(qubit) for qubit in qubits)
        if qubits is not None
        else tuple(sorted(int(qubit) for qubit in calibration.topology.nodes))
    )
    if not ordered_qubits:
        raise ValueError("No qubits are available for readout calibration.")

    programs = build_local_readout_calibration_programs(ordered_qubits)
    active_shots = max(1, int(shots))
    prepared_zero = _run_quil_program(qc, programs["zero"], shots=active_shots)
    prepared_one = _run_quil_program(qc, programs["one"], shots=active_shots)

    qubit_metrics = {int(qubit): dict(metrics) for qubit, metrics in calibration.qubit_metrics.items()}
    for bit_index, physical in enumerate(ordered_qubits):
        p10 = max(0.0, min(0.5 - EPS, _bit_marginal_probability(prepared_zero, bit_index, bit="1")))
        p01 = max(0.0, min(0.5 - EPS, _bit_marginal_probability(prepared_one, bit_index, bit="0")))
        metrics = dict(qubit_metrics.get(int(physical), {}))
        metrics["readout_p10"] = float(p10)
        metrics["readout_p01"] = float(p01)
        metrics["readout_fidelity"] = float(max(0.0, min(1.0, 1.0 - 0.5 * (p10 + p01))))
        qubit_metrics[int(physical)] = metrics

    return CalibrationData(
        path=calibration.path,
        topology=calibration.topology.copy(),
        node_cost=dict(calibration.node_cost),
        edge_cost=dict(calibration.edge_cost),
        qubit_metrics=qubit_metrics,
        pair_metrics={tuple(pair): dict(metrics) for pair, metrics in calibration.pair_metrics.items()},
        default_1q_duration_ns=calibration.default_1q_duration_ns,
        default_2q_duration_ns=calibration.default_2q_duration_ns,
    )


def compile_standard_qaoa_to_quil(
    pair_weights: Any,
    fields: Any,
    gammas: Any,
    betas: Any,
    *,
    config: CompilerConfig | None = None,
    initial_mapping: Mapping[int, int] | None = None,
) -> CompilationResult:
    """Build a standard QAOA circuit from solver tensors and compile it to Quil."""
    from ModularQaoaSetup.qaoa_solvers.baselines import build_standard_qaoa_circuit

    circuit = build_standard_qaoa_circuit(pair_weights, fields, gammas, betas)
    return compile_qiskit_to_quil(circuit, config=config, initial_mapping=initial_mapping)


def compile_warm_start_qaoa_to_quil(
    problem: Any,
    gammas: Any,
    betas: Any,
    *,
    config: CompilerConfig | None = None,
    initial_mapping: Mapping[int, int] | None = None,
) -> CompilationResult:
    """Build a warm-start QAOA circuit and compile it through the same Quil pipeline."""
    from ModularQaoaSetup.qaoa_solvers.core import build_warm_start_qaoa

    circuit = build_warm_start_qaoa(problem, gammas, betas)
    return compile_qiskit_to_quil(circuit, config=config, initial_mapping=initial_mapping)


def _resolve_pipeline_switches(config: CompilerConfig) -> PipelineSwitches:
    return PipelineSwitches(
        normalize_circuit=bool(config.switches.normalize_circuit),
        placement=bool(config.switches.placement),
        routing=bool(config.switches.routing),
        pulse_calibrations=bool(config.switches.pulse_calibrations and config.enable_pulse_calibrations),
        robust_overlays=bool(config.switches.robust_overlays),
        dynamical_decoupling=bool(config.switches.dynamical_decoupling and config.enable_dd),
        zne=bool(config.switches.zne),
        readout_mitigation=bool(config.switches.readout_mitigation),
    )


def _default_identity_mapping(
    logical_qubits: int,
    calibration: CalibrationData,
) -> dict[int, int]:
    physical_nodes = sorted(int(node) for node in calibration.topology.nodes)
    if logical_qubits > len(physical_nodes):
        raise ValueError(
            f"Requested {logical_qubits} logical qubits but calibration only exposes {len(physical_nodes)} physical qubits."
        )
    return {logical: physical_nodes[logical] for logical in range(logical_qubits)}


def build_local_folding_variants(
    result: CompilationResult,
    scale_factors: Sequence[int] = (1, 3, 5),
) -> dict[int, str]:
    """Generate locally folded Quil programs for zero-noise extrapolation."""
    variants: dict[int, str] = {}
    measured = [op for op in result.operations if op.name == "measure"]
    unitary = [op for op in result.operations if op.name != "measure"]

    for factor in scale_factors:
        if factor < 1 or factor % 2 == 0:
            raise ValueError("Scale factors must be positive odd integers.")

        folded: list[PhysicalOperation] = []
        if factor == 1:
            folded.extend(unitary)
        else:
            repeats = (factor - 1) // 2
            for op in unitary:
                folded.append(op)
                for _ in range(repeats):
                    folded.append(_inverse_operation(op))
                    folded.append(op)

        folded.extend(measured)
        moments = _schedule_operations(folded, result.calibration)
        body_quil = _emit_quil(moments, result.circuit.num_clbits)
        variants[int(factor)] = _compose_full_quil(result.pulse_calibration_quil, body_quil)

    return variants


def run_qcs_with_mitigation(
    qc: Any,
    result: CompilationResult,
    *,
    shots: int = 256,
    scale_factors: Sequence[int] = (1, 3, 5),
    objective: Callable[[Mapping[str, float]], float] | None = None,
    switches: PipelineSwitches | None = None,
    refresh_readout_calibration: bool = False,
    readout_calibration_shots: int | None = None,
) -> QCSExecutionResult:
    """Execute folded programs on QCS, then apply readout mitigation and Richardson extrapolation."""
    active_switches = result.pipeline_switches if switches is None else switches
    active_scale_factors = (
        (1,)
        if not active_switches.zne
        else _normalize_runtime_scale_factors(scale_factors)
    )
    folded_programs = build_local_folding_variants(result, scale_factors=active_scale_factors)
    objective_fn = _default_expectation_objective if objective is None else objective
    mitigation_calibration = result.calibration
    if active_switches.readout_mitigation and refresh_readout_calibration:
        logical_order = tuple(sorted(int(logical) for logical in result.final_logical_to_physical))
        physical_qubits = tuple(int(result.final_logical_to_physical[logical]) for logical in logical_order)
        mitigation_calibration = calibrate_local_readout_with_backend(
            qc,
            result.calibration,
            qubits=physical_qubits,
            shots=shots if readout_calibration_shots is None else readout_calibration_shots,
        )

    mitigated_expectations: dict[int, float] = {}
    raw_probabilities: dict[str, float] = {}
    readout_mitigated_probabilities: dict[str, float] = {}

    for scale in sorted(folded_programs):
        measured = _run_quil_program(qc, folded_programs[scale], shots=shots)
        if active_switches.readout_mitigation:
            corrected = mitigate_diagonal_readout(
                measured,
                mitigation_calibration,
                result.final_logical_to_physical,
            )
        else:
            corrected = dict(measured)
        mitigated_expectations[int(scale)] = float(objective_fn(corrected))
        if int(scale) == 1:
            raw_probabilities = dict(measured)
            readout_mitigated_probabilities = dict(corrected)

    mitigated = (
        richardson_extrapolate(mitigated_expectations)
        if active_switches.zne
        else float(mitigated_expectations[1])
    )
    return QCSExecutionResult(
        raw_probabilities=raw_probabilities,
        readout_mitigated_probabilities=readout_mitigated_probabilities,
        folded_expectations=mitigated_expectations,
        mitigated_expectation=float(mitigated),
        shots_per_scale=int(shots),
    )


def closed_loop_optimize_pulse(
    *,
    gate: str,
    qubits: Sequence[int],
    qc: Any | None = None,
    config: ClosedLoopOptimizationConfig | None = None,
    hardware_runner: Callable[[Any | None, PulseControlParameters, CalibrationData], float] | None = None,
) -> ClosedLoopOptimizationResult:
    """Tune a parameterized pulse family with a lightweight SPSA-style search loop."""
    config = ClosedLoopOptimizationConfig() if config is None else config
    calibration = load_calibration(
        config.calibration_path,
        quantum_processor=(getattr(qc, "quantum_processor", None) if qc is not None else None),
    )

    baseline = _normalize_pulse_parameters(
        _baseline_pulse_parameters(gate, qubits, control_segments=config.control_segments),
        config,
    )
    if hardware_runner is None and qc is None:
        raise ValueError("Provide either a QuantumComputer instance or a hardware_runner callback.")

    def evaluate(params: PulseControlParameters) -> float:
        normalized = _normalize_pulse_parameters(params, config)
        if hardware_runner is not None:
            return float(hardware_runner(qc, normalized, calibration))
        return float(_evaluate_pulse_parameters_on_qcs(qc, normalized, calibration, shots=config.shots_per_eval))

    baseline_score = evaluate(baseline)
    best = baseline
    best_score = baseline_score
    evaluations = 1
    budget = max(1, int(config.max_evaluations))

    current = baseline
    current_vector = _pulse_parameter_vector(current)
    rng = np.random.default_rng(int(config.spsa_seed))
    iteration = 0

    while current_vector.size > 0 and evaluations + 1 < budget:
        iteration += 1
        decay = (1.0 + float(iteration)) ** float(config.spsa_decay)
        perturbation = float(config.spsa_perturbation) / max(decay, 1.0)
        learning_rate = float(config.spsa_learning_rate) / max(math.sqrt(iteration), 1.0)
        direction = _sample_spsa_direction(current_vector.size, rng)

        plus = _vector_to_pulse_parameters(current, current_vector + perturbation * direction, config)
        minus = _vector_to_pulse_parameters(current, current_vector - perturbation * direction, config)
        plus_score = evaluate(plus)
        minus_score = evaluate(minus)
        evaluations += 2

        for candidate, score in ((plus, plus_score), (minus, minus_score)):
            if score > best_score + EPS:
                best = candidate
                best_score = score

        gradient = ((plus_score - minus_score) / max(2.0 * perturbation, EPS)) * direction
        gradient = np.clip(gradient, -float(config.gradient_clip), float(config.gradient_clip))
        current = _vector_to_pulse_parameters(current, current_vector + learning_rate * gradient, config)
        current_vector = _pulse_parameter_vector(current)

    if evaluations < budget and current != baseline:
        current_score = evaluate(current)
        evaluations += 1
        if current_score > best_score + EPS:
            best = current
            best_score = current_score

    return ClosedLoopOptimizationResult(
        baseline_parameters=baseline,
        best_parameters=best,
        baseline_score=float(baseline_score),
        best_score=float(best_score),
        evaluations=int(evaluations),
        calibration_quil=_emit_parameterized_pulse_calibration(best, calibration),
    )


def _compose_full_quil(pulse_calibration_quil: str, body_quil: str) -> str:
    sections = [section.strip() for section in (pulse_calibration_quil, body_quil) if section.strip()]
    return "\n\n".join(sections).strip() + "\n"


def _run_quil_program(
    qc: Any,
    quil_source: str,
    *,
    shots: int,
) -> dict[str, float]:
    from pyquil import Program

    program = Program(quil_source)
    program = program.copy()
    program.wrap_in_numshots_loop(int(shots))

    try:
        executable = qc.compile(program, to_native_gates=False, optimize=False)
    except TypeError:
        executable = qc.compile(program)

    result = qc.run(executable)
    readout = None
    if hasattr(result, "readout_data"):
        readout = result.readout_data.get("ro")
    if readout is None and hasattr(result, "get_register_map"):
        readout = result.get_register_map().get("ro")
    if readout is None:
        raise RuntimeError("Unable to extract 'ro' readout data from the QCS execution result.")

    shots_array = np.asarray(readout, dtype=int)
    if shots_array.ndim != 2:
        raise RuntimeError(f"Unexpected readout shape {shots_array.shape}; expected a 2D shot matrix.")

    return _probabilities_from_readout_matrix(shots_array)


def _probabilities_from_readout_matrix(readout: np.ndarray) -> dict[str, float]:
    if readout.size == 0:
        return {}
    counts: dict[str, int] = {}
    for row in readout:
        bitstring = "".join(str(int(bit)) for bit in row)
        counts[bitstring] = counts.get(bitstring, 0) + 1
    total = float(readout.shape[0])
    return {bitstring: count / total for bitstring, count in counts.items()}


def _bit_marginal_probability(
    probabilities: Mapping[str, float],
    bit_index: int,
    *,
    bit: str,
) -> float:
    if bit not in {"0", "1"}:
        raise ValueError("bit must be '0' or '1'.")
    total = 0.0
    for bitstring, probability in probabilities.items():
        if int(bit_index) < len(bitstring) and bitstring[int(bit_index)] == bit:
            total += float(probability)
    return float(total)


def _default_expectation_objective(probabilities: Mapping[str, float]) -> float:
    expectation = 0.0
    for bitstring, probability in probabilities.items():
        parity = -1.0 if sum(int(bit) for bit in bitstring) % 2 else 1.0
        expectation += parity * float(probability)
    return float(expectation)


def _normalize_runtime_scale_factors(scale_factors: Sequence[int]) -> tuple[int, ...]:
    normalized: list[int] = []
    seen: set[int] = set()
    for raw_scale in scale_factors:
        scale = int(raw_scale)
        if scale < 1 or scale % 2 == 0 or scale in seen:
            continue
        normalized.append(scale)
        seen.add(scale)
    if 1 not in seen:
        normalized.insert(0, 1)
    return tuple(normalized)


def _baseline_pulse_parameters(
    gate: str,
    qubits: Sequence[int],
    *,
    control_segments: int,
) -> PulseControlParameters:
    normalized = gate.strip().upper()
    qubit_tuple = tuple(int(qubit) for qubit in qubits)
    segment_count = max(3, int(control_segments))
    if normalized == "X":
        if len(qubit_tuple) != 1:
            raise ValueError("Single-qubit closed-loop optimization expects exactly one qubit.")
        drive = _default_drive_profile(segment_count, gate="X")
        return PulseControlParameters(
            gate="X",
            qubits=qubit_tuple,
            drive_segments=drive,
            quadrature_segments=_default_quadrature_profile(drive),
            frequency_segments_hz=tuple(0.0 for _ in range(segment_count)),
        )
    if normalized == "CZ":
        if len(qubit_tuple) != 2:
            raise ValueError("Two-qubit closed-loop optimization expects exactly two qubits.")
        return PulseControlParameters(
            gate="CZ",
            qubits=qubit_tuple,
            drive_segments=_default_drive_profile(segment_count, gate="CZ"),
            frequency_segments_hz=tuple(0.0 for _ in range(segment_count)),
        )
    raise ValueError(f"Unsupported closed-loop pulse target '{gate}'. Supported targets are 'X' and 'CZ'.")


def _normalize_pulse_parameters(
    params: PulseControlParameters,
    config: ClosedLoopOptimizationConfig,
) -> PulseControlParameters:
    gate = params.gate.strip().upper()
    qubits = tuple(int(qubit) for qubit in params.qubits)
    segment_count = max(3, int(config.control_segments))

    amplitude_scale = max(0.2, float(params.amplitude_scale))
    duration_scale = max(0.2, float(params.duration_scale))
    detuning_hz = max(-float(config.frequency_limit_hz), min(float(config.frequency_limit_hz), float(params.detuning_hz)))

    if gate == "X":
        drive_default = _default_drive_profile(segment_count, gate="X")
        drive = _coerce_segment_profile(
            params.drive_segments,
            drive_default,
            minimum=float(config.segment_min),
            maximum=float(config.segment_max),
        )
        quadrature = _coerce_segment_profile(
            params.quadrature_segments,
            _default_quadrature_profile(drive_default),
            minimum=-float(config.quadrature_limit),
            maximum=float(config.quadrature_limit),
        )
        frequency = _coerce_segment_profile(
            params.frequency_segments_hz,
            tuple(0.0 for _ in range(segment_count)),
            minimum=-float(config.frequency_limit_hz),
            maximum=float(config.frequency_limit_hz),
        )
        return PulseControlParameters(
            gate="X",
            qubits=qubits,
            amplitude_scale=amplitude_scale,
            duration_scale=duration_scale,
            drag_scale=max(0.1, float(params.drag_scale)),
            detuning_hz=detuning_hz,
            phase_shift=math.pi,
            drive_segments=drive,
            quadrature_segments=quadrature,
            frequency_segments_hz=frequency,
            echo_scale=1.0,
        )

    if gate == "CZ":
        drive = _coerce_segment_profile(
            params.drive_segments,
            _default_drive_profile(segment_count, gate="CZ"),
            minimum=float(config.segment_min),
            maximum=float(config.segment_max),
        )
        frequency = _coerce_segment_profile(
            params.frequency_segments_hz,
            tuple(0.0 for _ in range(segment_count)),
            minimum=-float(config.frequency_limit_hz),
            maximum=float(config.frequency_limit_hz),
        )
        phase_shift = max(-2.0 * math.pi, min(2.0 * math.pi, float(params.phase_shift)))
        echo_scale = max(0.5, min(1.5, float(params.echo_scale)))
        return PulseControlParameters(
            gate="CZ",
            qubits=qubits,
            amplitude_scale=amplitude_scale,
            duration_scale=duration_scale,
            drag_scale=1.0,
            detuning_hz=0.0,
            phase_shift=phase_shift,
            drive_segments=drive,
            quadrature_segments=(),
            frequency_segments_hz=frequency,
            echo_scale=echo_scale,
        )

    raise ValueError(f"Unsupported closed-loop pulse target '{params.gate}'. Supported targets are 'X' and 'CZ'.")


def _default_drive_profile(segment_count: int, *, gate: str) -> tuple[float, ...]:
    count = max(3, int(segment_count))
    samples: list[float] = []
    for index in range(count):
        position = index / max(count - 1, 1)
        base = math.sin(math.pi * position) ** 2
        if gate == "X":
            samples.append(float(0.35 + 0.65 * base))
        else:
            samples.append(float(0.55 + 0.45 * base))
    return tuple(samples)


def _default_quadrature_profile(drive: Sequence[float]) -> tuple[float, ...]:
    if not drive:
        return ()
    derivative: list[float] = []
    for index in range(len(drive)):
        left = float(drive[max(0, index - 1)])
        right = float(drive[min(len(drive) - 1, index + 1)])
        derivative.append(0.5 * (right - left))
    scale = max(max(abs(value) for value in derivative), EPS)
    return tuple(float(value / scale) for value in derivative)


def _coerce_segment_profile(
    values: Sequence[float],
    default: Sequence[float],
    *,
    minimum: float,
    maximum: float,
) -> tuple[float, ...]:
    payload = [float(value) for value in values]
    fallback = [float(value) for value in default]
    if not payload:
        payload = fallback
    elif len(payload) < len(fallback):
        payload.extend(fallback[len(payload) :])
    elif len(payload) > len(fallback):
        payload = payload[: len(fallback)]
    return tuple(max(minimum, min(maximum, float(value))) for value in payload)


def _pulse_parameter_vector(params: PulseControlParameters) -> np.ndarray:
    gate = params.gate.upper()
    values: list[float] = [
        float(params.amplitude_scale),
        float(params.duration_scale),
    ]
    if gate == "X":
        values.extend(
            [
                float(params.drag_scale),
                float(params.detuning_hz) / 1.0e7,
            ]
        )
        values.extend(float(value) for value in params.drive_segments)
        values.extend(float(value) for value in params.quadrature_segments)
        values.extend(float(value) / 1.0e7 for value in params.frequency_segments_hz)
    else:
        values.extend(
            [
                float(params.phase_shift) / math.pi,
                float(params.echo_scale),
            ]
        )
        values.extend(float(value) for value in params.drive_segments)
        values.extend(float(value) / 1.0e7 for value in params.frequency_segments_hz)
    return np.asarray(values, dtype=float)


def _vector_to_pulse_parameters(
    template: PulseControlParameters,
    vector: np.ndarray,
    config: ClosedLoopOptimizationConfig,
) -> PulseControlParameters:
    gate = template.gate.upper()
    values = np.asarray(vector, dtype=float).reshape(-1)
    cursor = 0

    amplitude_scale = float(values[cursor])
    cursor += 1
    duration_scale = float(values[cursor])
    cursor += 1

    if gate == "X":
        drive_count = len(template.drive_segments)
        quadrature_count = len(template.quadrature_segments)
        frequency_count = len(template.frequency_segments_hz)
        expected = 4 + drive_count + quadrature_count + frequency_count
        if values.size != expected:
            raise ValueError(f"Expected {expected} pulse parameters for X, received {values.size}.")

        drag_scale = float(values[cursor])
        cursor += 1
        detuning_hz = float(values[cursor]) * 1.0e7
        cursor += 1
        drive_segments = tuple(float(value) for value in values[cursor : cursor + drive_count])
        cursor += drive_count
        quadrature_segments = tuple(float(value) for value in values[cursor : cursor + quadrature_count])
        cursor += quadrature_count
        frequency_segments_hz = tuple(float(value) * 1.0e7 for value in values[cursor : cursor + frequency_count])
        return _normalize_pulse_parameters(
            PulseControlParameters(
                gate="X",
                qubits=template.qubits,
                amplitude_scale=amplitude_scale,
                duration_scale=duration_scale,
                drag_scale=drag_scale,
                detuning_hz=detuning_hz,
                phase_shift=math.pi,
                drive_segments=drive_segments,
                quadrature_segments=quadrature_segments,
                frequency_segments_hz=frequency_segments_hz,
                echo_scale=1.0,
            ),
            config,
        )

    drive_count = len(template.drive_segments)
    frequency_count = len(template.frequency_segments_hz)
    expected = 4 + drive_count + frequency_count
    if values.size != expected:
        raise ValueError(f"Expected {expected} pulse parameters for CZ, received {values.size}.")

    phase_shift = float(values[cursor]) * math.pi
    cursor += 1
    echo_scale = float(values[cursor])
    cursor += 1
    drive_segments = tuple(float(value) for value in values[cursor : cursor + drive_count])
    cursor += drive_count
    frequency_segments_hz = tuple(float(value) * 1.0e7 for value in values[cursor : cursor + frequency_count])
    return _normalize_pulse_parameters(
        PulseControlParameters(
            gate="CZ",
            qubits=template.qubits,
            amplitude_scale=amplitude_scale,
            duration_scale=duration_scale,
            drag_scale=1.0,
            detuning_hz=0.0,
            phase_shift=phase_shift,
            drive_segments=drive_segments,
            quadrature_segments=(),
            frequency_segments_hz=frequency_segments_hz,
            echo_scale=echo_scale,
        ),
        config,
    )


def _sample_spsa_direction(size: int, rng: np.random.Generator) -> np.ndarray:
    return (rng.integers(0, 2, size=size, dtype=np.int64) * 2 - 1).astype(float)


def _evaluate_pulse_parameters_on_qcs(
    qc: Any,
    params: PulseControlParameters,
    calibration: CalibrationData,
    *,
    shots: int,
) -> float:
    calibration_quil = _emit_parameterized_pulse_calibration(params, calibration)
    gate = params.gate.upper()
    if gate == "X":
        qubit = params.qubits[0]
        body = "\n".join(
            [
                "DECLARE ro BIT[1]",
                f"X {qubit}",
                f"X {qubit}",
                f"MEASURE {qubit} ro[0]",
            ]
        )
        probabilities = _run_quil_program(qc, _compose_full_quil(calibration_quil, body), shots=shots)
        corrected = mitigate_diagonal_readout(probabilities, calibration, {0: qubit})
        return float(corrected.get("0", 0.0))

    left, right = params.qubits
    body = "\n".join(
        [
            "DECLARE ro BIT[2]",
            f"H {left}",
            f"H {right}",
            f"CZ {left} {right}",
            f"CZ {left} {right}",
            f"H {left}",
            f"H {right}",
            f"MEASURE {left} ro[0]",
            f"MEASURE {right} ro[1]",
        ]
    )
    probabilities = _run_quil_program(qc, _compose_full_quil(calibration_quil, body), shots=shots)
    corrected = mitigate_diagonal_readout(probabilities, calibration, {0: left, 1: right})
    return float(corrected.get("00", 0.0))


def _emit_parameterized_pulse_calibration(
    params: PulseControlParameters,
    calibration: CalibrationData,
) -> str:
    gate = params.gate.upper()
    if gate == "X":
        qubit = int(params.qubits[0])
        metrics = calibration.qubit_metrics.get(qubit, {})
        segment_count = max(
            3,
            len(params.drive_segments),
            len(params.quadrature_segments),
            len(params.frequency_segments_hz),
            5,
        )
        drive_segments = tuple(float(value) for value in params.drive_segments) or _default_drive_profile(segment_count, gate="X")
        quadrature_segments = (
            tuple(float(value) for value in params.quadrature_segments)
            or _default_quadrature_profile(drive_segments)
        )
        frequency_segments_hz = tuple(float(value) for value in params.frequency_segments_hz) or tuple(
            0.0 for _ in range(len(drive_segments))
        )
        freq = (float(metrics.get("freq_ghz", 5.0)) * 1e9) + float(params.detuning_hz)
        duration = float(metrics.get("duration_ns", calibration.default_1q_duration_ns)) * float(params.duration_scale)
        anharmonicity_hz = float(metrics.get("anharmonicity_hz", DEFAULT_ANHARMONICITY_HZ))
        name = f"opt_x_{qubit}"
        lines = [
            f'DEFFRAME {qubit} "xy":',
            f"    INITIAL-FREQUENCY: {freq:.16g}",
            "",
        ]
        lines.extend(
            _emit_piecewise_drag_waveform_definition(
                name,
                duration_ns=duration,
                amplitude=float(params.amplitude_scale),
                anharmonicity_hz=anharmonicity_hz,
                drag_scale=float(params.drag_scale),
                drive_segments=drive_segments,
                quadrature_segments=quadrature_segments,
                frequency_segments_hz=frequency_segments_hz,
            )
        )
        lines.extend(
            [
                "",
                f"DEFCAL X {qubit}:",
                f'    PULSE {qubit} "xy" {name}',
                "",
            ]
        )
        return "\n".join(lines).strip() + "\n"

    left, right = (int(params.qubits[0]), int(params.qubits[1]))
    metrics = calibration.pair_metrics.get((left, right), calibration.pair_metrics.get((right, left), {}))
    segment_count = max(3, len(params.drive_segments), len(params.frequency_segments_hz), 5)
    drive_segments = tuple(float(value) for value in params.drive_segments) or _default_drive_profile(
        segment_count,
        gate="CZ",
    )
    frequency_segments_hz = tuple(float(value) for value in params.frequency_segments_hz) or tuple(
        0.0 for _ in range(len(drive_segments))
    )
    freq = max(abs(float(metrics.get("zz_mhz", 400.0))) * 1e6, 1.0e8)
    duration = float(metrics.get("duration_ns", calibration.default_2q_duration_ns)) * float(params.duration_scale)
    first_half_name = f"opt_cz_{left}_{right}_a"
    second_half_name = f"opt_cz_{left}_{right}_b"
    lines = [
        f'DEFFRAME {left} {right} "cz":',
        f"    INITIAL-FREQUENCY: {freq:.16g}",
        "",
    ]
    lines.extend(
        _emit_piecewise_coupler_waveform_definition(
            first_half_name,
            duration_ns=max(duration / 2.0, 1.0),
            amplitude=float(params.amplitude_scale),
            drive_segments=drive_segments,
            frequency_segments_hz=frequency_segments_hz,
        )
    )
    lines.extend(
        [
            "",
        ]
    )
    lines.extend(
        _emit_piecewise_coupler_waveform_definition(
            second_half_name,
            duration_ns=max(duration / 2.0, 1.0),
            amplitude=float(params.amplitude_scale) * float(params.echo_scale),
            drive_segments=tuple(reversed(drive_segments)),
            frequency_segments_hz=tuple(-value for value in reversed(frequency_segments_hz)),
        )
    )
    lines.extend(
        [
            "",
            f"DEFCAL CZ {left} {right}:",
            f"    FENCE {left} {right}",
            f'    PULSE {left} {right} "cz" {first_half_name}',
            f"    X {right}",
            f'    SHIFT-PHASE {left} {right} "cz" {float(params.phase_shift):.16g}',
            f'    PULSE {left} {right} "cz" {second_half_name}',
            f"    X {right}",
            f'    SHIFT-PHASE {left} {right} "cz" {-float(params.phase_shift):.16g}',
            f"    FENCE {left} {right}",
            "",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def _emit_piecewise_drag_waveform_definition(
    name: str,
    *,
    duration_ns: float,
    amplitude: float,
    anharmonicity_hz: float,
    drag_scale: float,
    drive_segments: Sequence[float],
    quadrature_segments: Sequence[float],
    frequency_segments_hz: Sequence[float],
) -> list[str]:
    sample_count = max(16, min(128, int(round(max(duration_ns, 1.0) / 4.0))))
    sigma = max(sample_count / 6.0, 1.0)
    center = (sample_count - 1) / 2.0
    duration_s = max(float(duration_ns), 1.0) * 1.0e-9
    dt = duration_s / max(sample_count, 1)
    beta = float(drag_scale) / max(2.0 * math.pi * float(anharmonicity_hz) * duration_s, EPS)
    phase_accumulator = 0.0
    samples: list[str] = []

    for index in range(sample_count):
        position = index / max(sample_count - 1, 1)
        t = (index - center) / sigma
        base_drive = float(amplitude) * math.exp(-0.5 * t * t)
        drive_scale = _interpolate_segments(drive_segments, position)
        quadrature_scale = _interpolate_segments(quadrature_segments, position)
        local_detuning_hz = _interpolate_segments(frequency_segments_hz, position)
        derivative = -beta * t * base_drive
        i_value = base_drive * drive_scale
        q_value = derivative + 0.25 * base_drive * quadrature_scale
        phase_accumulator += 2.0 * math.pi * local_detuning_hz * dt
        cos_phase = math.cos(phase_accumulator)
        sin_phase = math.sin(phase_accumulator)
        real_value = i_value * cos_phase - q_value * sin_phase
        imag_value = i_value * sin_phase + q_value * cos_phase
        samples.append(_format_complex_sample(real_value, imag_value))

    return [
        f"DEFWAVEFORM {name}:",
        f"    {', '.join(samples)}",
    ]


def _emit_piecewise_coupler_waveform_definition(
    name: str,
    *,
    duration_ns: float,
    amplitude: float,
    drive_segments: Sequence[float],
    frequency_segments_hz: Sequence[float],
) -> list[str]:
    sample_count = max(8, min(128, int(round(max(duration_ns, 1.0) / 4.0))))
    duration_s = max(float(duration_ns), 1.0) * 1.0e-9
    dt = duration_s / max(sample_count, 1)
    phase_accumulator = 0.0
    samples: list[str] = []

    for index in range(sample_count):
        position = index / max(sample_count - 1, 1)
        base = float(amplitude) * math.sin(math.pi * position) ** 2
        envelope = base * _interpolate_segments(drive_segments, position)
        local_detuning_hz = _interpolate_segments(frequency_segments_hz, position)
        phase_accumulator += 2.0 * math.pi * local_detuning_hz * dt
        real_value = envelope * math.cos(phase_accumulator)
        imag_value = envelope * math.sin(phase_accumulator)
        samples.append(_format_complex_sample(real_value, imag_value))

    return [
        f"DEFWAVEFORM {name}:",
        f"    {', '.join(samples)}",
    ]


def _interpolate_segments(values: Sequence[float], position: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    clamped = min(1.0, max(0.0, float(position)))
    scaled = clamped * (len(values) - 1)
    left_index = int(math.floor(scaled))
    right_index = min(left_index + 1, len(values) - 1)
    weight = scaled - left_index
    left = float(values[left_index])
    right = float(values[right_index])
    return (1.0 - weight) * left + weight * right


def _format_complex_sample(real_value: float, imag_value: float) -> str:
    if abs(imag_value) < 1e-12:
        return f"{real_value:.12g}"
    if imag_value >= 0.0:
        return f"{real_value:.12g}+{imag_value:.12g}i"
    return f"{real_value:.12g}{imag_value:.12g}i"

def _emit_pulse_calibrations(
    operations: Sequence[PhysicalOperation],
    calibration: CalibrationData,
    *,
    enable_robust_overlays: bool = True,
) -> str:
    used_qubits = sorted({int(qubit) for op in operations for qubit in op.qubits})
    used_edges = sorted(
        {
            tuple(sorted((int(op.qubits[0]), int(op.qubits[1]))))
            for op in operations
            if op.arity == 2 and op.name in {"cx", "cz", "swap"}
        }
    )
    if not used_qubits and not used_edges:
        return ""

    lines: list[str] = []

    for qubit in used_qubits:
        freq = float(calibration.qubit_metrics.get(qubit, {}).get("freq_ghz", 5.0) or 5.0) * 1e9
        duration = float(calibration.qubit_metrics.get(qubit, {}).get("duration_ns", calibration.default_1q_duration_ns))
        anharmonicity_hz = float(
            calibration.qubit_metrics.get(qubit, {}).get("anharmonicity_hz", DEFAULT_ANHARMONICITY_HZ)
        )
        pi_name = f"sq_{qubit}_pi"
        lines.append(f'DEFFRAME {qubit} "xy":')
        lines.append(f"    INITIAL-FREQUENCY: {freq:.16g}")
        lines.append("")
        lines.extend(
            _emit_drag_waveform_definition(
                pi_name,
                duration_ns=duration,
                amplitude=1.0,
                anharmonicity_hz=anharmonicity_hz,
                drag_scale=1.0,
            )
        )
        lines.append("")
        lines.append(f"DEFCAL X {qubit}:")
        lines.append(f'    PULSE {qubit} "xy" {pi_name}')
        lines.append("")
        lines.append(f"DEFCAL Y {qubit}:")
        lines.append(f'    SHIFT-PHASE {qubit} "xy" {math.pi / 2.0:.16g}')
        lines.append(f'    PULSE {qubit} "xy" {pi_name}')
        lines.append(f'    SHIFT-PHASE {qubit} "xy" {-math.pi / 2.0:.16g}')
        lines.append("")
        lines.append(f"DEFCAL Z {qubit}:")
        lines.append(f'    SHIFT-PHASE {qubit} "xy" {math.pi:.16g}')
        lines.append("")
        lines.append(f"DEFCAL H {qubit}:")
        lines.append(f"    RZ({math.pi / 2.0:.16g}) {qubit}")
        lines.append(f"    RX({math.pi / 2.0:.16g}) {qubit}")
        lines.append(f"    RZ({math.pi / 2.0:.16g}) {qubit}")
        lines.append("")
        lines.append(f"DEFCAL RX(%theta) {qubit}:")
        lines.append(f'    SET-SCALE {qubit} "xy" %theta/{math.pi:.16g}')
        lines.append(f'    PULSE {qubit} "xy" {pi_name}')
        lines.append(f'    SET-SCALE {qubit} "xy" 1.0')
        lines.append("")
        lines.append(f"DEFCAL RY(%theta) {qubit}:")
        lines.append(f'    SHIFT-PHASE {qubit} "xy" {math.pi / 2.0:.16g}')
        lines.append(f'    SET-SCALE {qubit} "xy" %theta/{math.pi:.16g}')
        lines.append(f'    PULSE {qubit} "xy" {pi_name}')
        lines.append(f'    SET-SCALE {qubit} "xy" 1.0')
        lines.append(f'    SHIFT-PHASE {qubit} "xy" {-math.pi / 2.0:.16g}')
        lines.append("")
        lines.append(f"DEFCAL RZ(%theta) {qubit}:")
        lines.append(f'    SHIFT-PHASE {qubit} "xy" %theta')
        lines.append("")

    for left, right in used_edges:
        pair_metric = calibration.pair_metrics.get((left, right), {})
        zz_mhz = float(pair_metric.get("zz_mhz", 400.0) or 400.0)
        freq = max(abs(zz_mhz) * 1e6, 1.0e8)
        duration = float(pair_metric.get("duration_ns", calibration.default_2q_duration_ns))
        fidelity = float(pair_metric.get("fidelity", 1.0))
        full_name = f"cz_{left}_{right}_full"
        half_name = f"cz_{left}_{right}_half"
        lines.append(f'DEFFRAME {left} {right} "cz":')
        lines.append(f"    INITIAL-FREQUENCY: {freq:.16g}")
        lines.append("")
        lines.extend(_emit_smooth_real_waveform_definition(full_name, duration_ns=duration, amplitude=1.0))
        lines.append("")
        lines.extend(
            _emit_smooth_real_waveform_definition(
                half_name,
                duration_ns=max(duration / 2.0, 1.0),
                amplitude=0.5,
            )
        )
        lines.append("")

        robust = enable_robust_overlays and fidelity < 0.97
        lines.append(f"DEFCAL CZ {left} {right}:")
        lines.append(f"    FENCE {left} {right}")
        if robust:
            lines.append(f'    PULSE {left} {right} "cz" {half_name}')
            lines.append(f"    X {right}")
            lines.append(f'    SHIFT-PHASE {left} {right} "cz" {math.pi:.16g}')
            lines.append(f'    PULSE {left} {right} "cz" {half_name}')
            lines.append(f"    X {right}")
            lines.append(f'    SHIFT-PHASE {left} {right} "cz" {-math.pi:.16g}')
        else:
            lines.append(f'    PULSE {left} {right} "cz" {full_name}')
        lines.append(f"    FENCE {left} {right}")
        lines.append("")

        for control, target in ((left, right), (right, left)):
            lines.append(f"DEFCAL CNOT {control} {target}:")
            lines.append(f"    H {target}")
            lines.append(f"    CZ {left} {right}")
            lines.append(f"    H {target}")
            lines.append("")

            lines.append(f"DEFCAL SWAP {control} {target}:")
            lines.append(f"    CNOT {control} {target}")
            lines.append(f"    CNOT {target} {control}")
            lines.append(f"    CNOT {control} {target}")
            lines.append("")

    return "\n".join(lines).strip() + "\n"


def _normalize_column_name(name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    normalized = normalized.replace("q_", "q")
    normalized = normalized.replace("1_q", "1q")
    normalized = normalized.replace("2_q", "2q")
    return normalized


def _coerce_metric(row: Mapping[str, Any], *keys: str, default: float) -> float:
    for key in keys:
        value = row.get(key)
        if value is None or pd.isna(value):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return float(default)


def _optional_metric(row: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = row.get(key)
        if value is None or pd.isna(value):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _parse_pair(raw: str) -> tuple[int, int] | None:
    match = re.match(r"^\s*(\d+)\s*[-:]\s*(\d+)\s*$", raw)
    if match is None:
        return None
    a, b = int(match.group(1)), int(match.group(2))
    return (a, b)


def _normalize_circuit(circuit: QuantumCircuit, *, optimization_level: int) -> QuantumCircuit:
    basis = ["h", "x", "y", "z", "rx", "ry", "rz", "cx", "cz", "swap"]
    return transpile(circuit, basis_gates=basis, optimization_level=optimization_level)


def _extract_logical_operations(circuit: QuantumCircuit) -> list[LogicalOperation]:
    operations: list[LogicalOperation] = []
    for entry in circuit.data:
        op = entry.operation
        name = op.name.lower()
        if name in {"barrier", "id"}:
            continue

        qubits = tuple(int(circuit.find_bit(qubit).index) for qubit in entry.qubits)
        clbits = tuple(int(circuit.find_bit(clbit).index) for clbit in entry.clbits)
        params = tuple(float(param) for param in op.params)

        if name not in {"h", "x", "y", "z", "rx", "ry", "rz", "cx", "cz", "swap", "measure"}:
            raise ValueError(
                f"Unsupported Qiskit instruction '{op.name}' after normalization. "
                "The circuit could not be reduced to the supported Rigetti translation basis."
            )

        operations.append(LogicalOperation(name=name, qubits=qubits, params=params, clbits=clbits))

    return operations


def _validated_initial_mapping(
    initial_mapping: Mapping[int, int],
    logical_qubits: int,
    calibration: CalibrationData,
) -> dict[int, int]:
    mapping = {int(logical): int(physical) for logical, physical in initial_mapping.items()}
    expected = set(range(logical_qubits))
    if set(mapping) != expected:
        raise ValueError(
            f"Initial mapping must cover exactly logical qubits {sorted(expected)}; received {sorted(mapping)}."
        )

    used = set(mapping.values())
    if len(used) != logical_qubits:
        raise ValueError("Initial mapping must assign each logical qubit to a distinct physical qubit.")
    unknown = used.difference(int(node) for node in calibration.topology.nodes)
    if unknown:
        raise ValueError(f"Initial mapping references unknown physical qubits: {sorted(unknown)}")
    return mapping


def _candidate_initial_mappings(
    logical_qubits: int,
    logical_ops: Sequence[LogicalOperation],
    calibration: CalibrationData,
    *,
    trials: int,
) -> list[dict[int, int]]:
    if logical_qubits == 0:
        return [{}]

    physical_nodes = sorted(int(node) for node in calibration.topology.nodes)
    if logical_qubits > len(physical_nodes):
        raise ValueError(
            f"Requested {logical_qubits} logical qubits but calibration only exposes {len(physical_nodes)} physical qubits."
        )

    interaction_graph = _build_interaction_graph(logical_qubits, logical_ops)
    seeds = sorted(physical_nodes, key=lambda node: calibration.node_cost.get(node, 1.0))[
        : min(max(1, trials * 2), len(physical_nodes))
    ]

    candidates: list[dict[int, int]] = []
    seen: set[tuple[int, ...]] = set()
    for seed in seeds:
        subgraph = _greedy_connected_subgraph(seed, logical_qubits, calibration)
        if len(subgraph) < logical_qubits:
            continue
        candidate = _assign_logical_to_physical(interaction_graph, tuple(subgraph), calibration)
        key = tuple(int(candidate[index]) for index in range(logical_qubits))
        if key in seen:
            continue
        seen.add(key)
        candidates.append(candidate)

    if candidates:
        candidates.sort(key=lambda mapping: _mapping_score(interaction_graph, mapping, calibration))
        return candidates[: max(1, trials)]

    fallback_nodes = sorted(physical_nodes, key=lambda node: calibration.node_cost.get(node, 1.0))[:logical_qubits]
    return [{logical: physical for logical, physical in enumerate(fallback_nodes)}]


def _build_interaction_graph(
    logical_qubits: int,
    logical_ops: Sequence[LogicalOperation],
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(logical_qubits))
    for op in logical_ops:
        if op.arity != 2 or op.name == "measure":
            continue
        a, b = op.qubits
        weight = graph[a][b]["weight"] + 1.0 if graph.has_edge(a, b) else 1.0
        graph.add_edge(a, b, weight=float(weight))
    return graph


def _greedy_connected_subgraph(seed: int, size: int, calibration: CalibrationData) -> list[int]:
    selected = [int(seed)]
    selected_set = {int(seed)}

    while len(selected) < size:
        frontier: dict[int, float] = {}
        for node in selected:
            for neighbor in calibration.topology.neighbors(node):
                neighbor = int(neighbor)
                if neighbor in selected_set:
                    continue
                affinity = sum(
                    1.0 / (1.0 + calibration.edge_cost.get((neighbor, current), 1.0))
                    for current in selected
                    if calibration.topology.has_edge(neighbor, current)
                )
                score = calibration.node_cost.get(neighbor, 1.0) - 0.5 * affinity
                frontier[neighbor] = min(frontier.get(neighbor, math.inf), score)

        if not frontier:
            break

        chosen = min(frontier, key=frontier.get)
        selected.append(chosen)
        selected_set.add(chosen)

    return selected


def _assign_logical_to_physical(
    interaction_graph: nx.Graph,
    physical_nodes: Sequence[int],
    calibration: CalibrationData,
) -> dict[int, int]:
    logical_order = sorted(
        (int(node) for node in interaction_graph.nodes),
        key=lambda node: (
            -interaction_graph.degree(node, weight="weight"),
            -interaction_graph.degree(node),
            int(node),
        ),
    )

    mapping: dict[int, int] = {}
    remaining = set(int(node) for node in physical_nodes)
    cached_paths: dict[tuple[int, int], float] = {}

    for logical in logical_order:
        best_physical = None
        best_score = math.inf
        for physical in sorted(remaining):
            score = calibration.node_cost.get(physical, 1.0)
            for placed_logical, placed_physical in mapping.items():
                weight = 0.0
                if interaction_graph.has_edge(logical, placed_logical):
                    weight = float(interaction_graph[logical][placed_logical]["weight"])
                if weight <= 0.0:
                    continue
                key = tuple(sorted((physical, placed_physical)))
                if key not in cached_paths:
                    cached_paths[key] = _shortest_path_cost(calibration, key[0], key[1])
                score += weight * cached_paths[key]

            if score < best_score:
                best_physical = physical
                best_score = score

        if best_physical is None:
            raise RuntimeError("Unable to assign a physical qubit for the current logical qubit.")

        mapping[logical] = best_physical
        remaining.remove(best_physical)

    return mapping


def _mapping_score(
    interaction_graph: nx.Graph,
    mapping: Mapping[int, int],
    calibration: CalibrationData,
) -> float:
    score = sum(calibration.node_cost.get(int(physical), 1.0) for physical in mapping.values())
    for left, right, data in interaction_graph.edges(data=True):
        score += float(data.get("weight", 1.0)) * _shortest_path_cost(
            calibration,
            int(mapping[int(left)]),
            int(mapping[int(right)]),
        )
    return float(score)


def _shortest_path_cost(calibration: CalibrationData, start: int, end: int) -> float:
    if start == end:
        return 0.0
    try:
        return float(nx.shortest_path_length(calibration.topology, start, end, weight="cost"))
    except nx.NetworkXNoPath:
        return 1e6


def _best_route_path(calibration: CalibrationData, start: int, end: int) -> list[int]:
    if start == end:
        return [start]

    try:
        candidates = nx.shortest_simple_paths(calibration.topology, start, end, weight="cost")
    except nx.NetworkXNoPath as exc:
        raise ValueError(f"No routing path exists between physical qubits {start} and {end}.") from exc

    best_path: list[int] | None = None
    best_score = math.inf
    for index, path in enumerate(candidates):
        if index >= 3:
            break
        path = [int(node) for node in path]
        edge_penalty = sum(
            calibration.edge_cost.get((int(left), int(right)), 1.0)
            for left, right in zip(path[:-1], path[1:])
        )
        swap_penalty = max(0, len(path) - 2) * 2.0
        score = edge_penalty + swap_penalty
        if score < best_score:
            best_score = score
            best_path = path

    if best_path is None:
        raise ValueError(f"No candidate route was produced between physical qubits {start} and {end}.")
    return best_path


def _route_logical_operations(
    logical_ops: Sequence[LogicalOperation],
    initial_mapping: Mapping[int, int],
    calibration: CalibrationData,
    *,
    enable_routing: bool = True,
) -> tuple[list[PhysicalOperation], dict[int, int]]:
    logical_to_physical = {int(logical): int(physical) for logical, physical in initial_mapping.items()}
    physical_to_logical = {physical: logical for logical, physical in logical_to_physical.items()}
    routed: list[PhysicalOperation] = []

    for op in logical_ops:
        if op.name == "measure":
            logical = int(op.qubits[0])
            physical = logical_to_physical[logical]
            routed.append(
                PhysicalOperation(
                    name="measure",
                    qubits=(physical,),
                    clbits=op.clbits,
                    source="logical",
                )
            )
            continue

        if op.arity == 1:
            logical = int(op.qubits[0])
            physical = logical_to_physical[logical]
            routed.append(
                PhysicalOperation(
                    name=op.name,
                    qubits=(physical,),
                    params=op.params,
                    source="logical",
                )
            )
            continue

        if op.arity != 2:
            raise ValueError(f"Unsupported routed arity {op.arity} for operation '{op.name}'.")

        left_logical, right_logical = (int(op.qubits[0]), int(op.qubits[1]))
        left_physical = logical_to_physical[left_logical]
        right_physical = logical_to_physical[right_logical]

        if not calibration.topology.has_edge(left_physical, right_physical):
            if not enable_routing:
                raise ValueError(
                    "Routing stage is disabled, but the current placement requires a non-adjacent "
                    f"{op.name.upper()} between physical qubits {left_physical} and {right_physical}."
                )
            path = _best_route_path(calibration, left_physical, right_physical)
            for start, end in zip(path[:-2], path[1:-1]):
                routed.append(PhysicalOperation(name="swap", qubits=(int(start), int(end)), source="routing"))
                _apply_physical_swap(int(start), int(end), logical_to_physical, physical_to_logical)

        routed.append(
            PhysicalOperation(
                name=op.name,
                qubits=(
                    logical_to_physical[left_logical],
                    logical_to_physical[right_logical],
                ),
                params=op.params,
                source="logical",
            )
        )
        if op.name == "swap":
            _apply_physical_swap(
                logical_to_physical[left_logical],
                logical_to_physical[right_logical],
                logical_to_physical,
                physical_to_logical,
            )

    return routed, dict(logical_to_physical)


def _apply_physical_swap(
    left: int,
    right: int,
    logical_to_physical: dict[int, int],
    physical_to_logical: dict[int, int],
) -> None:
    left_logical = physical_to_logical.get(left)
    right_logical = physical_to_logical.get(right)

    if left_logical is None and right_logical is None:
        return
    if left_logical is None:
        physical_to_logical[left] = right_logical  # type: ignore[index]
        del physical_to_logical[right]
        logical_to_physical[right_logical] = left  # type: ignore[index]
        return
    if right_logical is None:
        physical_to_logical[right] = left_logical
        del physical_to_logical[left]
        logical_to_physical[left_logical] = right
        return

    physical_to_logical[left], physical_to_logical[right] = right_logical, left_logical
    logical_to_physical[left_logical], logical_to_physical[right_logical] = right, left


def _schedule_operations(
    operations: Sequence[PhysicalOperation],
    calibration: CalibrationData,
) -> list[list[PhysicalOperation]]:
    next_free: dict[int, int] = {}
    moments: list[list[PhysicalOperation]] = []

    for op in operations:
        start = 0
        for qubit in op.qubits:
            start = max(start, next_free.get(int(qubit), 0))
        duration = _duration_cycles(op, calibration)
        for cycle in range(start, start + duration):
            while len(moments) <= cycle:
                moments.append([])
        moments[start].append(op)
        for qubit in op.qubits:
            next_free[int(qubit)] = start + duration

    return moments


def _build_occupancy(
    moments: Sequence[Sequence[PhysicalOperation]],
    calibration: CalibrationData,
) -> tuple[list[int], dict[int, list[bool]]]:
    used_qubits = sorted({int(qubit) for moment in moments for op in moment for qubit in op.qubits})
    occupied: dict[int, list[bool]] = {qubit: [False] * len(moments) for qubit in used_qubits}

    for cycle, moment in enumerate(moments):
        for op in moment:
            for offset in range(_duration_cycles(op, calibration)):
                active_cycle = cycle + offset
                for qubit in op.qubits:
                    occupied[int(qubit)][active_cycle] = True

    return used_qubits, occupied


def _score_physical_schedule(
    moments: Sequence[Sequence[PhysicalOperation]],
    calibration: CalibrationData,
) -> float:
    instruction_cost = 0.0
    for moment in moments:
        for op in moment:
            instruction_cost += 1.0
            if op.arity == 2:
                edge_penalty = calibration.edge_cost.get((int(op.qubits[0]), int(op.qubits[1])), 1.0)
                instruction_cost += edge_penalty
                if op.source == "routing":
                    instruction_cost += 2.0 * edge_penalty

    _, occupied = _build_occupancy(moments, calibration)
    idle_penalty = 0.0
    for qubit, busy in occupied.items():
        t1_us = float(calibration.qubit_metrics.get(qubit, {}).get("t1", 20.0))
        idle_run = 0
        for flag in list(busy) + [True]:
            if not flag:
                idle_run += 1
                continue
            if idle_run:
                idle_ns = idle_run * calibration.default_1q_duration_ns
                idle_penalty += idle_ns / max(t1_us * 1000.0, EPS)
                idle_run = 0

    return float(instruction_cost + idle_penalty)


def _operation_duration_ns(op: PhysicalOperation, calibration: CalibrationData) -> float:
    if op.name == "measure":
        return float(calibration.default_1q_duration_ns)
    if op.name in {"x", "y", "rx", "ry", "h"}:
        qubit = int(op.qubits[0])
        return _effective_1q_pulse_duration_ns(calibration, qubit)
    if op.name in {"z", "rz"}:
        return float(calibration.default_1q_duration_ns)
    if op.name == "cz":
        edge = tuple(sorted((int(op.qubits[0]), int(op.qubits[1]))))
        metric = calibration.pair_metrics.get(edge, calibration.pair_metrics.get((edge[1], edge[0]), {}))
        fidelity = float(metric.get("fidelity", 1.0))
        if fidelity < 0.97:
            right = int(op.qubits[1])
            return (
                2.0 * _effective_2q_pulse_duration_ns(calibration, edge, half_pulse=True)
                + 2.0 * _effective_1q_pulse_duration_ns(calibration, right)
            )
        return _effective_2q_pulse_duration_ns(calibration, edge, half_pulse=False)
    if op.name == "cx":
        control, target = (int(op.qubits[0]), int(op.qubits[1]))
        return (
            2.0 * _effective_1q_pulse_duration_ns(calibration, target)
            + _operation_duration_ns(PhysicalOperation(name="cz", qubits=(control, target)), calibration)
        )
    if op.name == "swap":
        left, right = (int(op.qubits[0]), int(op.qubits[1]))
        cx_duration = _operation_duration_ns(PhysicalOperation(name="cx", qubits=(left, right)), calibration)
        return 3.0 * cx_duration
    qubit = int(op.qubits[0])
    return float(calibration.qubit_metrics.get(qubit, {}).get("duration_ns", calibration.default_1q_duration_ns))


def _effective_1q_pulse_duration_ns(calibration: CalibrationData, qubit: int) -> float:
    raw_duration = float(calibration.qubit_metrics.get(int(qubit), {}).get("duration_ns", calibration.default_1q_duration_ns))
    sample_count = max(16, min(128, int(round(max(raw_duration, 1.0) / 4.0))))
    return float(sample_count * 4.0)


def _effective_2q_pulse_duration_ns(
    calibration: CalibrationData,
    edge: tuple[int, int],
    *,
    half_pulse: bool,
) -> float:
    metric = calibration.pair_metrics.get(edge, calibration.pair_metrics.get((edge[1], edge[0]), {}))
    raw_duration = float(metric.get("duration_ns", calibration.default_2q_duration_ns))
    effective_duration = max(raw_duration / 2.0, 1.0) if half_pulse else max(raw_duration, 1.0)
    sample_count = max(8, min(128, int(round(effective_duration / 4.0))))
    return float(sample_count * 4.0)


def _duration_cycles(op: PhysicalOperation, calibration: CalibrationData) -> int:
    quantum_ns = max(float(calibration.default_1q_duration_ns), EPS)
    duration_ns = _operation_duration_ns(op, calibration)
    return max(1, int(math.ceil(duration_ns / quantum_ns)))


def _insert_xy4_dd(
    moments: list[list[PhysicalOperation]],
    calibration: CalibrationData,
    *,
    idle_cycles: int,
    risk_threshold: float,
    synchronize_pairs: bool,
) -> None:
    if not moments:
        return

    used_qubits, occupied = _build_occupancy(moments, calibration)
    if not used_qubits:
        return

    if synchronize_pairs:
        risky_edges = sorted(
            (
                (int(left), int(right))
                for left, right in calibration.topology.edges()
                if left in occupied and right in occupied
            ),
            key=lambda edge: calibration.edge_cost.get(edge, 0.0),
            reverse=True,
        )

        for left, right in risky_edges:
            edge_risk = calibration.edge_cost.get((left, right), 0.0)
            if edge_risk + EPS < risk_threshold:
                continue
            shared_free = [not occupied[left][index] and not occupied[right][index] for index in range(len(moments))]
            start = 0
            while start < len(shared_free):
                if not shared_free[start]:
                    start += 1
                    continue
                end = start
                while end + 1 < len(shared_free) and shared_free[end + 1]:
                    end += 1
                if (end - start + 1) >= idle_cycles:
                    slots = _spread_slots(start, end, count=4)
                    if len(slots) == 4:
                        for name, slot in zip(("x", "y", "x", "y"), slots):
                            left_op = PhysicalOperation(name=name, qubits=(left,), source="dd")
                            right_op = PhysicalOperation(name=name, qubits=(right,), source="dd")
                            if not _can_place_operation_at_cycle(left_op, slot, occupied, calibration):
                                continue
                            if not _can_place_operation_at_cycle(right_op, slot, occupied, calibration):
                                continue
                            moments[slot].append(left_op)
                            moments[slot].append(right_op)
                            _mark_occupied(left_op, slot, occupied, calibration)
                            _mark_occupied(right_op, slot, occupied, calibration)
                start = end + 1

    for qubit in used_qubits:
        if calibration.crosstalk_risk(qubit) + EPS < risk_threshold:
            continue
        busy = occupied[qubit]
        busy_cycles = [index for index, flag in enumerate(busy) if flag]
        if len(busy_cycles) < 2:
            continue

        for left, right in zip(busy_cycles, busy_cycles[1:]):
            gap_start = left + 1
            gap_end = right - 1
            gap = gap_end - gap_start + 1
            if gap < idle_cycles:
                continue
            slots = _spread_slots(gap_start, gap_end, count=4)
            if len(slots) < 4:
                continue
            for name, slot in zip(("x", "y", "x", "y"), slots):
                dd_op = PhysicalOperation(name=name, qubits=(qubit,), source="dd")
                if not _can_place_operation_at_cycle(dd_op, slot, occupied, calibration):
                    continue
                moments[slot].append(dd_op)
                _mark_occupied(dd_op, slot, occupied, calibration)

    for moment in moments:
        moment.sort(key=lambda op: (op.source != "dd", op.arity, op.qubits, op.name))


def _can_place_operation_at_cycle(
    op: PhysicalOperation,
    start_cycle: int,
    occupied: Mapping[int, list[bool]],
    calibration: CalibrationData,
) -> bool:
    duration = _duration_cycles(op, calibration)
    for qubit in op.qubits:
        timeline = occupied.get(int(qubit))
        if timeline is None:
            return False
        if start_cycle + duration > len(timeline):
            return False
        if any(timeline[start_cycle + offset] for offset in range(duration)):
            return False
    return True


def _mark_occupied(
    op: PhysicalOperation,
    start_cycle: int,
    occupied: dict[int, list[bool]],
    calibration: CalibrationData,
) -> None:
    duration = _duration_cycles(op, calibration)
    for qubit in op.qubits:
        timeline = occupied[int(qubit)]
        for offset in range(duration):
            timeline[start_cycle + offset] = True


def _recommend_calibration_overlays(
    operations: Sequence[PhysicalOperation],
    calibration: CalibrationData,
    *,
    fidelity_threshold: float = 0.97,
    enable_robust_overlays: bool = True,
) -> tuple[str, ...]:
    recommendations: list[str] = []
    seen_edges: set[tuple[int, int]] = set()
    for op in operations:
        if op.arity != 2 or op.name == "swap":
            continue
        edge = tuple(sorted((int(op.qubits[0]), int(op.qubits[1]))))
        if edge in seen_edges:
            continue
        seen_edges.add(edge)
        fidelity = float(calibration.pair_metrics.get(edge, {}).get("fidelity", 1.0))
        if fidelity >= fidelity_threshold:
            continue
        if enable_robust_overlays:
            recommendations.append(
                (
                    f"Edge {edge[0]}-{edge[1]} has 2Q fidelity {fidelity:.4f}; "
                    "the emitted Quil-T DEFCAL uses an echoed robust gate variant on this coupling, "
                    "but it is still a heuristic overlay rather than a hardware-identified optimal-control pulse."
                )
            )
        else:
            recommendations.append(
                (
                    f"Edge {edge[0]}-{edge[1]} has 2Q fidelity {fidelity:.4f}; "
                    "the robust overlay stage is disabled, so the emitted Quil-T DEFCAL uses the nominal CZ pulse."
                )
            )
    return tuple(recommendations)


def _spread_slots(start: int, end: int, *, count: int) -> list[int]:
    if start > end:
        return []
    width = end - start + 1
    if width < count:
        return []
    if count == 1:
        return [start]

    step = max(1, (width - 1) // (count - 1))
    slots = [start + step * index for index in range(count - 1)]
    slots.append(end)
    unique = []
    for slot in slots:
        clamped = min(end, max(start, slot))
        if clamped not in unique:
            unique.append(clamped)
    return unique if len(unique) == count else []


def _emit_drag_waveform_definition(
    name: str,
    *,
    duration_ns: float,
    amplitude: float,
    anharmonicity_hz: float,
    drag_scale: float,
) -> list[str]:
    sample_count = max(16, min(128, int(round(max(duration_ns, 1.0) / 4.0))))
    sigma = max(sample_count / 6.0, 1.0)
    center = (sample_count - 1) / 2.0
    beta = float(drag_scale) / max(2.0 * math.pi * anharmonicity_hz * (duration_ns * 1e-9), EPS)
    samples: list[str] = []
    for index in range(sample_count):
        t = (index - center) / sigma
        gaussian = amplitude * math.exp(-0.5 * t * t)
        derivative = -beta * t * gaussian
        if abs(derivative) < 1e-12:
            samples.append(f"{gaussian:.12g}")
        elif derivative >= 0.0:
            samples.append(f"{gaussian:.12g}+{derivative:.12g}i")
        else:
            samples.append(f"{gaussian:.12g}{derivative:.12g}i")
    entries = ", ".join(samples)
    return [
        f"DEFWAVEFORM {name}:",
        f"    {entries}",
    ]


def _emit_smooth_real_waveform_definition(
    name: str,
    *,
    duration_ns: float,
    amplitude: float,
) -> list[str]:
    sample_count = max(8, min(128, int(round(max(duration_ns, 1.0) / 4.0))))
    samples = []
    for index in range(sample_count):
        phase = math.pi * index / max(sample_count - 1, 1)
        value = amplitude * math.sin(phase) ** 2
        samples.append(f"{value:.12g}")
    entries = ", ".join(samples)
    return [
        f"DEFWAVEFORM {name}:",
        f"    {entries}",
    ]


def _emit_quil(moments: Sequence[Sequence[PhysicalOperation]], num_clbits: int) -> str:
    lines: list[str] = []
    if num_clbits > 0:
        lines.append(f"DECLARE ro BIT[{num_clbits}]")

    for moment in moments:
        if not moment:
            continue
        if lines:
            lines.append("")
        for op in moment:
            if op.source == "dd":
                lines.append(f"# DD {op.name.upper()} on q{op.qubits[0]}")
            lines.extend(_emit_instruction(op))

    return "\n".join(lines).strip() + "\n"


def _emit_instruction(op: PhysicalOperation) -> list[str]:
    q = op.qubits
    if op.name == "h":
        return [f"H {q[0]}"]
    if op.name == "x":
        return [f"X {q[0]}"]
    if op.name == "y":
        return [f"Y {q[0]}"]
    if op.name == "z":
        return [f"Z {q[0]}"]
    if op.name == "rx":
        return [f"RX({_format_angle(op.params[0])}) {q[0]}"]
    if op.name == "ry":
        return [f"RY({_format_angle(op.params[0])}) {q[0]}"]
    if op.name == "rz":
        return [f"RZ({_format_angle(op.params[0])}) {q[0]}"]
    if op.name == "cx":
        return [f"CNOT {q[0]} {q[1]}"]
    if op.name == "cz":
        return [f"CZ {q[0]} {q[1]}"]
    if op.name == "swap":
        return [f"SWAP {q[0]} {q[1]}"]
    if op.name == "measure":
        target = op.clbits[0] if op.clbits else 0
        return [f"MEASURE {q[0]} ro[{target}]"]
    raise ValueError(f"Unsupported Quil emission for operation '{op.name}'.")


def _format_angle(value: float) -> str:
    return f"{float(value):.16g}"


def _inverse_operation(op: PhysicalOperation) -> PhysicalOperation:
    if op.name in {"h", "x", "y", "z", "cx", "cz", "swap"}:
        return op
    if op.name in {"rx", "ry", "rz"}:
        return PhysicalOperation(
            name=op.name,
            qubits=op.qubits,
            params=(-float(op.params[0]),),
            clbits=op.clbits,
            source=op.source,
        )
    raise ValueError(f"Cannot build a folded inverse for operation '{op.name}'.")
