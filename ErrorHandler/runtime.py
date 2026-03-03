"""Runtime convenience wrappers around the ErrorHandler compilation pipeline.

The ``SmokyQuartz`` class packages calibration loading, Quil compilation, and
mitigated execution/simulation behind a small stateful interface that is easier
to use from notebooks and higher-level solver code.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import QuantumCircuit

from .compiler import (
    CalibrationData,
    CompilationResult,
    CompilerConfig,
    PipelineSwitches,
    QCSExecutionResult,
    compile_qiskit_to_quil,
    load_calibration,
    richardson_extrapolate,
    run_qcs_with_mitigation,
)


@dataclass(frozen=True)
class SmokyQuartzExecutionConfig:
    """Execution settings derived from instance defaults plus per-call overrides."""
    enabled: bool
    compiler_config: CompilerConfig
    shots: int
    scale_factors: tuple[int, ...]
    legacy_runtime_mitigation: bool = False


@dataclass
class SmokyQuartz:
    """Small convenience wrapper for compiling and executing circuits against one calibration."""
    calibration: CalibrationData
    compiler_config: CompilerConfig
    default_shots: int = 256
    default_scale_factors: tuple[int, ...] = (1, 3, 5)

    def __init__(
        self,
        calibration: CalibrationData | str | Path | None = None,
        *,
        compiler_config: CompilerConfig | None = None,
        default_shots: int = 256,
        default_scale_factors: tuple[int, ...] = (1, 3, 5),
    ) -> None:
        base_config = CompilerConfig() if compiler_config is None else compiler_config
        resolved_calibration = self._coerce_calibration(calibration, base_config=base_config)
        self.calibration = resolved_calibration
        self.compiler_config = replace(base_config, calibration_path=resolved_calibration.path)
        self.default_shots = max(1, int(default_shots))
        self.default_scale_factors = _normalize_scale_factors(default_scale_factors)

    @classmethod
    def from_calibration_path(
        cls,
        calibration_path: str | Path,
        *,
        compiler_config: CompilerConfig | None = None,
        default_shots: int = 256,
        default_scale_factors: tuple[int, ...] = (1, 3, 5),
    ) -> "SmokyQuartz":
        """Construct an instance directly from a calibration CSV path."""
        return cls(
            calibration=Path(calibration_path),
            compiler_config=compiler_config,
            default_shots=default_shots,
            default_scale_factors=default_scale_factors,
        )

    def refresh_calibration(
        self,
        calibration: CalibrationData | str | Path | None = None,
    ) -> CalibrationData:
        """Reload calibration data and update the stored compiler configuration."""
        resolved = self._coerce_calibration(calibration, base_config=self.compiler_config)
        self.calibration = resolved
        self.compiler_config = replace(self.compiler_config, calibration_path=resolved.path)
        return resolved

    def compile(
        self,
        circuit: QuantumCircuit,
        *,
        initial_mapping: dict[int, int] | None = None,
        qc: Any | None = None,
        compiler_config: CompilerConfig | None = None,
    ) -> CompilationResult:
        """Compile a Qiskit circuit using the instance calibration by default."""
        return compile_qiskit_to_quil(
            circuit,
            config=self._resolve_compiler_config(compiler_config),
            initial_mapping=initial_mapping,
            qc=qc,
        )

    def run_with_backend(
        self,
        qc: Any,
        circuit_or_compilation: QuantumCircuit | CompilationResult,
        *,
        shots: int | None = None,
        scale_factors: tuple[int, ...] | None = None,
        refresh_readout_calibration: bool = False,
        readout_calibration_shots: int | None = None,
        switches: PipelineSwitches | None = None,
    ) -> QCSExecutionResult:
        """Compile if needed, execute folded variants, and return mitigated runtime data."""
        compilation = (
            circuit_or_compilation
            if isinstance(circuit_or_compilation, CompilationResult)
            else self.compile(circuit_or_compilation, qc=qc)
        )
        return run_qcs_with_mitigation(
            qc,
            compilation,
            shots=self.default_shots if shots is None else max(1, int(shots)),
            scale_factors=self.default_scale_factors if scale_factors is None else _normalize_scale_factors(scale_factors),
            refresh_readout_calibration=refresh_readout_calibration,
            readout_calibration_shots=(
                None if readout_calibration_shots is None else max(1, int(readout_calibration_shots))
            ),
            switches=switches,
        )

    def build_execution_config(
        self,
        *,
        enabled: bool = True,
        shots: int | None = None,
        scale_factors: tuple[int, ...] | None = None,
        legacy_runtime_mitigation: bool = False,
    ) -> SmokyQuartzExecutionConfig:
        """Merge default runtime settings with per-call execution overrides."""
        return SmokyQuartzExecutionConfig(
            enabled=enabled,
            compiler_config=self.compiler_config,
            shots=self.default_shots if shots is None else max(1, int(shots)),
            scale_factors=(
                self.default_scale_factors
                if scale_factors is None
                else _normalize_scale_factors(scale_factors)
            ),
            legacy_runtime_mitigation=legacy_runtime_mitigation,
        )

    def simulate_qaoa_probabilities(
        self,
        circuit: QuantumCircuit,
        *,
        error_model: str = "ideal",
        layout_seed: int = 0,
        shots: int | None = None,
        scale_factors: tuple[int, ...] | None = None,
        execution_enabled: bool = True,
        legacy_runtime_mitigation: bool = False,
        measurement_qubits: int | None = None,
        sampling_enabled: bool = False,
    ) -> tuple[np.ndarray, float]:
        """Return the measured probability vector and its sampling uncertainty estimate."""
        runtime_config = self.build_execution_config(
            enabled=execution_enabled,
            shots=shots,
            scale_factors=scale_factors,
            legacy_runtime_mitigation=legacy_runtime_mitigation,
        )
        measured_qubits = _normalize_measurement_qubit_count(circuit, measurement_qubits)
        probabilities, sampled = _simulate_qiskit_probabilities(
            circuit,
            calibration=self.calibration,
            error_model=str(error_model),
            layout_seed=int(layout_seed),
            shots=runtime_config.shots,
            measurement_qubits=measured_qubits,
            scale_factors=runtime_config.scale_factors,
            mitigation_enabled=bool(runtime_config.enabled),
            sampling_enabled=bool(sampling_enabled),
        )
        if sampled:
            return probabilities, _max_probability_standard_error(probabilities, runtime_config.shots)
        return probabilities, 0.0

    def simulate_qaoa_expectation(
        self,
        circuit: QuantumCircuit,
        observable_values: np.ndarray,
        *,
        error_model: str = "ideal",
        layout_seed: int = 0,
        shots: int | None = None,
        scale_factors: tuple[int, ...] | None = None,
        execution_enabled: bool = True,
        legacy_runtime_mitigation: bool = False,
        measurement_qubits: int | None = None,
        sampling_enabled: bool = False,
    ) -> tuple[float, float]:
        """Evaluate an observable against the simulated probability vector for a QAOA circuit."""
        runtime_config = self.build_execution_config(
            enabled=execution_enabled,
            shots=shots,
            scale_factors=scale_factors,
            legacy_runtime_mitigation=legacy_runtime_mitigation,
        )
        measured_qubits = _normalize_measurement_qubit_count(circuit, measurement_qubits)
        probabilities, sampled = _simulate_qiskit_probabilities(
            circuit,
            calibration=self.calibration,
            error_model=str(error_model),
            layout_seed=int(layout_seed),
            shots=runtime_config.shots,
            measurement_qubits=measured_qubits,
            scale_factors=runtime_config.scale_factors,
            mitigation_enabled=bool(runtime_config.enabled),
            sampling_enabled=bool(sampling_enabled),
        )
        values = np.asarray(observable_values, dtype=float).reshape(-1)
        if values.size != probabilities.size:
            raise ValueError(
                "observable_values must have one entry per measured basis state "
                f"({probabilities.size} expected, received {values.size})."
            )
        expectation = float(values @ probabilities)
        if not sampled:
            return expectation, 0.0

        second_moment = float((values * values) @ probabilities)
        variance = max(0.0, second_moment - expectation * expectation)
        return expectation, float(np.sqrt(variance / runtime_config.shots))

    def _resolve_compiler_config(self, compiler_config: CompilerConfig | None) -> CompilerConfig:
        active_config = self.compiler_config if compiler_config is None else compiler_config
        return replace(active_config, calibration_path=self.calibration.path)

    @staticmethod
    def _coerce_calibration(
        calibration: CalibrationData | str | Path | None,
        *,
        base_config: CompilerConfig,
    ) -> CalibrationData:
        if isinstance(calibration, CalibrationData):
            return calibration

        calibration_path = base_config.calibration_path if calibration is None else Path(calibration)
        return load_calibration(calibration_path)


def _normalize_scale_factors(scale_factors: tuple[int, ...]) -> tuple[int, ...]:
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


def _normalize_measurement_qubit_count(
    circuit: QuantumCircuit,
    measurement_qubits: int | None,
) -> int:
    count = int(circuit.num_qubits) if measurement_qubits is None else max(0, int(measurement_qubits))
    return min(count, int(circuit.num_qubits))


def _simulate_qiskit_probabilities(
    circuit: QuantumCircuit,
    *,
    calibration: CalibrationData,
    error_model: str,
    layout_seed: int,
    shots: int,
    measurement_qubits: int,
    scale_factors: tuple[int, ...],
    mitigation_enabled: bool,
    sampling_enabled: bool,
) -> tuple[np.ndarray, bool]:
    if measurement_qubits <= 0:
        return np.array([1.0], dtype=float), False

    normalized_error_model = str(error_model).strip().lower()
    del sampling_enabled
    noise_model = None
    effective_mitigation = False
    if normalized_error_model not in {"", "ideal", "none", "statevector"}:
        noise_model = _build_qiskit_noise_model(
            calibration,
            qubit_count=int(circuit.num_qubits),
            measurement_qubits=measurement_qubits,
        )
        effective_mitigation = bool(mitigation_enabled)

    probabilities = _simulate_mitigated_qiskit_probabilities(
        circuit,
        noise_model=noise_model,
        calibration=calibration,
        layout_seed=layout_seed,
        shots=shots,
        measurement_qubits=measurement_qubits,
        scale_factors=_normalize_scale_factors(scale_factors),
        mitigation_enabled=effective_mitigation,
    )
    return probabilities, True


def _simulate_mitigated_qiskit_probabilities(
    circuit: QuantumCircuit,
    *,
    noise_model,
    calibration: CalibrationData,
    layout_seed: int,
    shots: int,
    measurement_qubits: int,
    scale_factors: tuple[int, ...],
    mitigation_enabled: bool,
) -> np.ndarray:
    active_scales = (1,) if not mitigation_enabled else _normalize_scale_factors(scale_factors)
    per_scale: dict[int, np.ndarray] = {}

    for scale in active_scales:
        folded = _fold_circuit_locally(circuit, int(scale))
        observed_probabilities = _simulate_aer_counts_probabilities(
            folded,
            noise_model=noise_model,
            layout_seed=layout_seed + int(scale),
            shots=shots,
            measurement_qubits=measurement_qubits,
        )
        corrected = (
            _mitigate_readout_vector(
                observed_probabilities,
                calibration=calibration,
                measurement_qubits=measurement_qubits,
            )
            if mitigation_enabled
            else observed_probabilities
        )
        per_scale[int(scale)] = corrected

    if len(per_scale) == 1:
        return next(iter(per_scale.values()))
    return _richardson_extrapolate_vector(per_scale)


def _simulate_aer_counts_probabilities(
    circuit: QuantumCircuit,
    *,
    noise_model=None,
    layout_seed: int,
    shots: int,
    measurement_qubits: int,
) -> np.ndarray:
    from qiskit import ClassicalRegister, transpile
    from qiskit_aer import AerSimulator

    sampled_circuit = circuit.remove_final_measurements(inplace=False)
    sampled_circuit = sampled_circuit.copy()
    classical = ClassicalRegister(int(measurement_qubits), "meas")
    sampled_circuit.add_register(classical)
    sampled_circuit.measure(
        list(range(int(measurement_qubits))),
        list(range(int(measurement_qubits))),
    )

    simulator = AerSimulator(
        method="automatic",
        noise_model=noise_model,
        seed_simulator=int(layout_seed),
    )
    compiled = transpile(
        sampled_circuit,
        simulator,
        seed_transpiler=int(layout_seed),
        optimization_level=0,
    )
    result = simulator.run(compiled, shots=max(1, int(shots))).result()
    counts = result.get_counts()
    if isinstance(counts, list):
        counts = counts[0]
    return _probability_vector_from_counts(dict(counts), measurement_qubits)


def _build_qiskit_noise_model(
    calibration: CalibrationData,
    *,
    qubit_count: int,
    measurement_qubits: int,
):
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

    noise_model = NoiseModel()
    one_qubit_gates = ("id", "x", "y", "z", "h", "sx", "rx", "ry", "rz")
    two_qubit_gates = ("cx", "cz", "swap", "ecr", "rzz")

    for qubit in range(qubit_count):
        metrics = calibration.qubit_metrics.get(qubit, {})
        one_qubit_fidelity = float(metrics.get("one_qubit_fidelity", 1.0))
        one_qubit_error_rate = max(0.0, min(1.0, 1.0 - one_qubit_fidelity))
        if one_qubit_error_rate > 0.0:
            error = depolarizing_error(one_qubit_error_rate, 1)
            for gate in one_qubit_gates:
                noise_model.add_quantum_error(error, gate, [qubit])

        if qubit < measurement_qubits:
            readout_fidelity = float(metrics.get("readout_fidelity", 1.0))
            flip_probability = max(0.0, min(0.5 - 1e-12, 1.0 - readout_fidelity))
            if flip_probability > 0.0:
                readout_error = ReadoutError(
                    [
                        [1.0 - flip_probability, flip_probability],
                        [flip_probability, 1.0 - flip_probability],
                    ]
                )
                noise_model.add_readout_error(readout_error, [qubit])

    seen_pairs: set[tuple[int, int]] = set()
    for left, right in calibration.topology.edges():
        pair = tuple(sorted((int(left), int(right))))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        if pair[0] >= qubit_count or pair[1] >= qubit_count:
            continue

        metrics = calibration.pair_metrics.get(pair, calibration.pair_metrics.get((pair[1], pair[0]), {}))
        two_qubit_fidelity = float(metrics.get("fidelity", 1.0))
        two_qubit_error_rate = max(0.0, min(1.0, 1.0 - two_qubit_fidelity))
        if two_qubit_error_rate <= 0.0:
            continue

        error = depolarizing_error(two_qubit_error_rate, 2)
        for gate in two_qubit_gates:
            noise_model.add_quantum_error(error, gate, [pair[0], pair[1]])
            noise_model.add_quantum_error(error, gate, [pair[1], pair[0]])

    return noise_model


def _probability_vector_from_counts(
    counts: dict[str, int],
    measurement_qubits: int,
) -> np.ndarray:
    probabilities = np.zeros(1 << measurement_qubits, dtype=float)
    total_shots = float(sum(int(count) for count in counts.values()))
    if total_shots <= 0.0:
        return probabilities

    for bitstring, count in counts.items():
        normalized_bits = str(bitstring).replace(" ", "")
        basis_state = int(normalized_bits[::-1], 2)
        probabilities[basis_state] += float(count) / total_shots
    return probabilities




def _fold_circuit_locally(circuit: QuantumCircuit, scale_factor: int) -> QuantumCircuit:
    scale = max(1, int(scale_factor))
    if scale == 1:
        return circuit.copy()
    if scale % 2 == 0:
        raise ValueError("scale_factor must be an odd positive integer.")

    repeats = (scale - 1) // 2
    base = circuit.remove_final_measurements(inplace=False)
    folded = QuantumCircuit(*base.qregs, *base.cregs, name=base.name)
    folded.global_phase = base.global_phase

    for instruction in base.data:
        operation = instruction.operation
        qubits = instruction.qubits
        clbits = instruction.clbits
        if str(operation.name).lower() == "barrier":
            folded.append(operation, qubits, clbits)
            continue

        folded.append(operation, qubits, clbits)
        inverse_operation = operation.inverse()
        for _ in range(repeats):
            folded.append(inverse_operation, qubits, clbits)
            folded.append(operation, qubits, clbits)

    return folded

def _mitigate_readout_vector(
    probabilities: np.ndarray,
    *,
    calibration: CalibrationData,
    measurement_qubits: int,
) -> np.ndarray:
    mitigated = _apply_factored_readout_map(
        probabilities,
        calibration=calibration,
        measurement_qubits=measurement_qubits,
        inverse=True,
    )
    mitigated = np.clip(mitigated, 0.0, None)
    total = float(np.sum(mitigated))
    if total > 0.0:
        mitigated /= total
    return mitigated


def _apply_factored_readout_map(
    probabilities: np.ndarray,
    *,
    calibration: CalibrationData,
    measurement_qubits: int,
    inverse: bool,
) -> np.ndarray:
    if measurement_qubits <= 0:
        return np.array([1.0], dtype=float)

    transformed = np.asarray(probabilities, dtype=float).reshape((2,) * measurement_qubits)
    for logical in range(measurement_qubits):
        metrics = calibration.qubit_metrics.get(int(logical), {})
        fidelity = float(metrics.get("readout_fidelity", 0.95))
        flip_10 = max(0.0, min(0.5 - 1e-12, float(metrics.get("readout_p10", 1.0 - fidelity))))
        flip_01 = max(0.0, min(0.5 - 1e-12, float(metrics.get("readout_p01", 1.0 - fidelity))))
        local = np.array(
            [
                [1.0 - flip_10, flip_01],
                [flip_10, 1.0 - flip_01],
            ],
            dtype=float,
        )
        transform = np.linalg.inv(local) if inverse else local
        axis = measurement_qubits - 1 - logical
        working = np.moveaxis(transformed, axis, 0)
        working = np.tensordot(transform, working, axes=([1], [0]))
        transformed = np.moveaxis(working, 0, axis)
    return transformed.reshape(-1)

def _richardson_extrapolate_vector(observations: dict[int, np.ndarray]) -> np.ndarray:
    if not observations:
        raise ValueError("At least one observation vector is required.")
    ordered_scales = sorted(int(scale) for scale in observations)
    reference_shape = observations[ordered_scales[0]].shape
    if any(observations[scale].shape != reference_shape for scale in ordered_scales):
        raise ValueError("All observation vectors must have the same shape.")

    extrapolated = np.zeros(reference_shape, dtype=float)
    for index in range(extrapolated.size):
        values = {
            scale: float(observations[scale].reshape(-1)[index])
            for scale in ordered_scales
        }
        extrapolated.reshape(-1)[index] = richardson_extrapolate(values)

    extrapolated = np.clip(extrapolated, 0.0, None)
    total = float(np.sum(extrapolated))
    if total > 0.0:
        extrapolated /= total
    return extrapolated


def _max_probability_standard_error(
    probabilities: np.ndarray,
    shots: int,
) -> float:
    if shots <= 0 or probabilities.size == 0:
        return 0.0
    variances = probabilities * (1.0 - probabilities)
    return float(np.max(np.sqrt(np.maximum(variances, 0.0) / float(shots))))
