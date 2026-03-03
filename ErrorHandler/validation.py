"""Validation and lightweight analog-analysis helpers for generated Quil programs.

These helpers inspect compiler output after the fact: they check whether pulse
calibration templates resolve correctly, whether folded programs preserve the
expected structure, and whether the emitted schedule looks sane under a compact
surrogate noise model.
"""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

import ErrorHandler.compiler as _compiler
from .compiler import CalibrationData, CompilationResult, PhysicalOperation, build_local_folding_variants


SAMPLE_PERIOD_NS = 4.0
_ANGLE_GATES = {"RX", "RY", "RZ"}
_SINGLE_QUBIT_GATES = {"H", "X", "Y", "Z"} | _ANGLE_GATES
_TWO_QUBIT_GATES = {"CNOT", "CZ", "SWAP"}
EPS = 1e-9


@dataclass(frozen=True)
class PulseEvent:
    """One pulse-level event reconstructed from a parsed Quil calibration block."""
    frame: str
    waveform: str
    qubits: tuple[int, ...]
    source_gate: str
    moment_index: int
    start_ns: float
    duration_ns: float
    amplitude: float


@dataclass(frozen=True)
class CalibrationRoundTripReport:
    """Validation report for resolving gates against the generated calibration definitions."""
    parsed_successfully: bool
    resolved_gate_count: int
    missing_calibrations: tuple[str, ...]
    missing_waveforms: tuple[str, ...]
    missing_frames: tuple[str, ...]
    unresolved_symbols: tuple[str, ...]
    template_violations: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return (
            self.parsed_successfully
            and not self.missing_calibrations
            and not self.missing_waveforms
            and not self.missing_frames
            and not self.unresolved_symbols
            and not self.template_violations
        )


@dataclass(frozen=True)
class PulseScheduleReport:
    """Summary of a reconstructed pulse schedule and any detected timing issues."""
    total_makespan_ns: float
    per_qubit_idle_ns: dict[int, float]
    simultaneous_pulse_events: int
    frame_overlaps: tuple[str, ...]
    negative_waits: tuple[str, ...]
    amplitude_violations: tuple[str, ...]
    alignment_violations: tuple[str, ...]
    events: tuple[PulseEvent, ...]

    @property
    def is_valid(self) -> bool:
        return (
            not self.frame_overlaps
            and not self.negative_waits
            and not self.amplitude_violations
            and not self.alignment_violations
        )


@dataclass(frozen=True)
class FoldingValidationReport:
    """Checks that a folded program matches the expected local-folding structure."""
    scale_factor: int
    measurement_mapping_preserved: bool
    structural_mismatches: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return self.measurement_mapping_preserved and not self.structural_mismatches


@dataclass(frozen=True)
class AnalogSimulationReport:
    """Compact gate-level analog surrogate results for one compiled program."""
    ideal_probabilities: dict[str, float]
    noisy_probabilities: dict[str, float]
    readout_probabilities: dict[str, float]
    ideal_expectation: float
    noisy_expectation: float
    readout_expectation: float
    zz_phase_error: float
    total_duration_ns: float


@dataclass(frozen=True)
class _WaveformDefinition:
    name: str
    samples: tuple[complex, ...]

    @property
    def duration_ns(self) -> float:
        return float(len(self.samples) * SAMPLE_PERIOD_NS)

    @property
    def peak_amplitude(self) -> float:
        if not self.samples:
            return 0.0
        return max(abs(sample) for sample in self.samples)


@dataclass(frozen=True)
class _CalibrationDefinition:
    name: str
    qubits: tuple[int, ...]
    param_names: tuple[str, ...]
    body_lines: tuple[str, ...]


@dataclass(frozen=True)
class _GateInvocation:
    name: str
    qubits: tuple[int, ...]
    params: tuple[str, ...]
    raw_line: str


@dataclass(frozen=True)
class _ProgramModel:
    frames: dict[tuple[tuple[int, ...], str], dict[str, float]]
    waveforms: dict[str, _WaveformDefinition]
    calibrations: dict[tuple[str, tuple[int, ...], int], _CalibrationDefinition]
    body_lines: tuple[str, ...]


class _SafeEvaluator(ast.NodeVisitor):
    def visit_Expression(self, node: ast.Expression) -> float:
        return float(self.visit(node.body))

    def visit_BinOp(self, node: ast.BinOp) -> float:
        left = float(self.visit(node.left))
        right = float(self.visit(node.right))
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left**right
        raise ValueError(f"Unsupported operator {ast.dump(node.op)}")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> float:
        value = float(self.visit(node.operand))
        if isinstance(node.op, ast.UAdd):
            return value
        if isinstance(node.op, ast.USub):
            return -value
        raise ValueError(f"Unsupported unary operator {ast.dump(node.op)}")

    def visit_Name(self, node: ast.Name) -> float:
        if node.id == "pi":
            return float(math.pi)
        raise ValueError(f"Unsupported symbol {node.id}")

    def visit_Constant(self, node: ast.Constant) -> float:
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"Unsupported constant {node.value!r}")

    def generic_visit(self, node: ast.AST) -> float:
        raise ValueError(f"Unsupported expression {ast.dump(node)}")


def validate_calibration_program(
    result: CompilationResult,
    *,
    quil_source: str | None = None,
) -> CalibrationRoundTripReport:
    """Verify that emitted gates can be resolved against the generated pulse templates."""
    source = result.quil if quil_source is None else quil_source
    parsed_successfully = True
    try:
        from pyquil import Program

        Program(source)
    except Exception:
        parsed_successfully = False

    model = _parse_quil_program(source)
    missing_calibrations: list[str] = []
    missing_waveforms: list[str] = []
    missing_frames: list[str] = []
    unresolved_symbols: list[str] = []
    template_violations: list[str] = []
    resolved_gate_count = 0
    visited_blocks: set[tuple[str, tuple[int, ...], int]] = set()

    for line in model.body_lines:
        invocation = _parse_gate_invocation(line)
        if invocation is None or invocation.name == "MEASURE":
            continue
        resolved_gate_count += 1
        _validate_invocation(
            invocation,
            model,
            result.calibration,
            visited_blocks=visited_blocks,
            missing_calibrations=missing_calibrations,
            missing_waveforms=missing_waveforms,
            missing_frames=missing_frames,
            unresolved_symbols=unresolved_symbols,
            template_violations=template_violations,
            substitutions={},
        )

    return CalibrationRoundTripReport(
        parsed_successfully=parsed_successfully,
        resolved_gate_count=resolved_gate_count,
        missing_calibrations=tuple(sorted(set(missing_calibrations))),
        missing_waveforms=tuple(sorted(set(missing_waveforms))),
        missing_frames=tuple(sorted(set(missing_frames))),
        unresolved_symbols=tuple(sorted(set(unresolved_symbols))),
        template_violations=tuple(sorted(set(template_violations))),
    )


def analyze_pulse_schedule(
    result: CompilationResult,
    *,
    amplitude_limit: float = 2.0,
    alignment_tolerance_ns: float = 1e-6,
    sensitive_zz_mhz: float = 0.0,
) -> PulseScheduleReport:
    """Reconstruct a coarse pulse schedule and flag overlaps, amplitude, and alignment issues."""
    model = _parse_quil_program(result.quil)
    frame_available: dict[tuple[tuple[int, ...], str], float] = {
        frame_key: 0.0 for frame_key in model.frames
    }
    frame_scale: dict[tuple[tuple[int, ...], str], float] = {
        frame_key: 1.0 for frame_key in model.frames
    }

    events: list[PulseEvent] = []
    frame_overlaps: list[str] = []
    negative_waits: list[str] = []
    amplitude_violations: list[str] = []
    base_cycle_ns = float(result.calibration.default_1q_duration_ns)

    for moment_index, moment in enumerate(result.moments):
        moment_start_ns = float(moment_index * base_cycle_ns)
        for op in moment:
            invocation = _invocation_from_physical_operation(op)
            if invocation.name == "MEASURE":
                continue
            primary_frames = _frames_for_invocation(invocation, model)
            for frame_key in primary_frames:
                if frame_available.get(frame_key, 0.0) > moment_start_ns + alignment_tolerance_ns:
                    frame_overlaps.append(
                        (
                            f"Frame {_frame_label(frame_key)} is busy until "
                            f"{frame_available[frame_key]:.6f} ns but moment {moment_index} starts at "
                            f"{moment_start_ns:.6f} ns."
                        )
                    )
            _expand_invocation_to_events(
                invocation,
                model,
                start_ns=moment_start_ns,
                moment_index=moment_index,
                frame_available=frame_available,
                frame_scale=frame_scale,
                events=events,
                missing_calibrations=[],
                missing_waveforms=[],
                missing_frames=[],
                unresolved_symbols=[],
                substitutions={},
            )

    events.sort(key=lambda event: (event.start_ns, event.frame, event.waveform))

    for current in events:
        if current.start_ns + alignment_tolerance_ns < 0.0:
            negative_waits.append(f"Pulse {current.source_gate} starts at negative time {current.start_ns:.6f} ns.")
        if current.amplitude > amplitude_limit + EPS:
            amplitude_violations.append(
                (
                    f"Pulse {current.waveform} on {current.frame} reaches amplitude {current.amplitude:.6f}, "
                    f"exceeding the configured limit {amplitude_limit:.6f}."
                )
            )

    total_makespan_ns = 0.0
    if events:
        total_makespan_ns = max(event.start_ns + event.duration_ns for event in events)

    busy_per_qubit: dict[int, float] = {}
    for op in result.operations:
        duration_ns = _compiler._operation_duration_ns(op, result.calibration)
        for qubit in op.qubits:
            busy_per_qubit[int(qubit)] = busy_per_qubit.get(int(qubit), 0.0) + float(duration_ns)
    per_qubit_idle_ns = {
        qubit: max(0.0, total_makespan_ns - busy_per_qubit.get(qubit, 0.0))
        for qubit in sorted({int(qubit) for op in result.operations for qubit in op.qubits})
    }

    simultaneous = _max_simultaneous_pulse_events(events)
    alignment_violations = _alignment_violations(
        result,
        tolerance_ns=alignment_tolerance_ns,
        sensitive_zz_mhz=sensitive_zz_mhz,
    )

    return PulseScheduleReport(
        total_makespan_ns=float(total_makespan_ns),
        per_qubit_idle_ns=per_qubit_idle_ns,
        simultaneous_pulse_events=int(simultaneous),
        frame_overlaps=tuple(frame_overlaps),
        negative_waits=tuple(negative_waits),
        amplitude_violations=tuple(amplitude_violations),
        alignment_violations=tuple(alignment_violations),
        events=tuple(events),
    )


def validate_folded_program(
    result: CompilationResult,
    *,
    scale_factor: int = 3,
) -> FoldingValidationReport:
    """Confirm that a local-folded program preserves instruction order and measurements."""
    variants = build_local_folding_variants(result, scale_factors=(scale_factor,))
    source = variants[int(scale_factor)]
    body_lines = [
        line
        for line in _parse_quil_program(source).body_lines
        if line and not line.startswith("DECLARE")
    ]

    expected_ops: list[PhysicalOperation] = []
    unitary = [op for op in result.operations if op.name != "measure"]
    measured = [op for op in result.operations if op.name == "measure"]

    repeats = (int(scale_factor) - 1) // 2
    if int(scale_factor) == 1:
        expected_ops.extend(unitary)
    else:
        for op in unitary:
            expected_ops.append(op)
            for _ in range(repeats):
                expected_ops.append(_compiler._inverse_operation(op))
                expected_ops.append(op)
    expected_ops.extend(measured)

    expected_lines: list[str] = []
    for op in expected_ops:
        expected_lines.extend(_compiler._emit_instruction(op))

    structural_mismatches: list[str] = []
    if len(body_lines) != len(expected_lines):
        structural_mismatches.append(
            f"Folded body length {len(body_lines)} does not match expected {len(expected_lines)}."
        )
    for index, (actual, expected) in enumerate(zip(body_lines, expected_lines)):
        if actual != expected:
            structural_mismatches.append(
                f"Instruction {index} mismatch: expected '{expected}', received '{actual}'."
            )

    original_measure_lines = [
        line
        for op in measured
        for line in _compiler._emit_instruction(op)
    ]
    actual_measure_lines = [line for line in body_lines if line.startswith("MEASURE")]

    return FoldingValidationReport(
        scale_factor=int(scale_factor),
        measurement_mapping_preserved=(actual_measure_lines == original_measure_lines),
        structural_mismatches=tuple(structural_mismatches),
    )


def simulate_effective_pulse_dynamics(
    result: CompilationResult,
    *,
    include_zz: bool = True,
    include_readout: bool = True,
) -> AnalogSimulationReport:
    """Estimate ideal and noisy observables with a compact gate-duration surrogate model."""
    # This is a compact gate-level surrogate that uses compiled durations plus ZZ/T1/T2
    # channels. It is intentionally not a full waveform-resolved pulse simulation.
    noisy_with_zz = _simulate_density_matrix(result, include_zz=include_zz)
    ideal = _simulate_density_matrix(result, include_zz=False, include_decoherence=False)
    noisy_without_zz = _simulate_density_matrix(result, include_zz=False)

    ideal_probabilities = _measurement_probabilities(ideal, result)
    noisy_probabilities = _measurement_probabilities(noisy_with_zz, result)
    readout_probabilities = (
        _apply_readout_model(noisy_probabilities, result) if include_readout else dict(noisy_probabilities)
    )

    ideal_expectation = _parity_expectation(ideal_probabilities)
    noisy_expectation = _parity_expectation(noisy_probabilities)
    readout_expectation = _parity_expectation(readout_probabilities)
    zz_phase_error = abs(noisy_expectation - _parity_expectation(_measurement_probabilities(noisy_without_zz, result)))
    total_duration_ns = sum(
        max((_compiler._operation_duration_ns(op, result.calibration) for op in moment), default=0.0)
        for moment in result.moments
    )

    return AnalogSimulationReport(
        ideal_probabilities=ideal_probabilities,
        noisy_probabilities=noisy_probabilities,
        readout_probabilities=readout_probabilities,
        ideal_expectation=float(ideal_expectation),
        noisy_expectation=float(noisy_expectation),
        readout_expectation=float(readout_expectation),
        zz_phase_error=float(zz_phase_error),
        total_duration_ns=float(total_duration_ns),
    )


def _parse_quil_program(quil_source: str) -> _ProgramModel:
    frames: dict[tuple[tuple[int, ...], str], dict[str, float]] = {}
    waveforms: dict[str, _WaveformDefinition] = {}
    calibrations: dict[tuple[str, tuple[int, ...], int], _CalibrationDefinition] = {}
    body_lines: list[str] = []

    lines = [line.rstrip() for line in quil_source.splitlines()]
    index = 0

    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line.startswith("DEFFRAME"):
            targets, frame_name = _parse_frame_header(line)
            frames[(targets, frame_name)] = {}
            index += 1
            while index < len(lines) and lines[index].startswith(" "):
                key, value = _parse_frame_attribute(lines[index].strip())
                frames[(targets, frame_name)][key] = value
                index += 1
            continue
        if line.startswith("DEFWAVEFORM"):
            name = _parse_waveform_header(line)
            sample_lines: list[str] = []
            index += 1
            while index < len(lines) and lines[index].startswith(" "):
                sample_lines.append(lines[index].strip())
                index += 1
            waveforms[name] = _WaveformDefinition(name=name, samples=_parse_waveform_samples(sample_lines))
            continue
        if line.startswith("DEFCAL"):
            definition, index = _parse_calibration_definition(lines, index)
            calibrations[(definition.name, definition.qubits, len(definition.param_names))] = definition
            continue
        body_lines.append(line)
        index += 1

    return _ProgramModel(
        frames=frames,
        waveforms=waveforms,
        calibrations=calibrations,
        body_lines=tuple(body_lines),
    )


def _parse_frame_header(line: str) -> tuple[tuple[int, ...], str]:
    match = re.match(r'^DEFFRAME\s+(.+?)\s+"([^"]+)":$', line)
    if match is None:
        raise ValueError(f"Unrecognized DEFFRAME header: {line}")
    targets = tuple(int(token) for token in match.group(1).split())
    return targets, match.group(2)


def _parse_frame_attribute(line: str) -> tuple[str, float]:
    name, _, value = line.partition(":")
    return name.strip(), _evaluate_numeric_expression(value.strip())


def _parse_waveform_header(line: str) -> str:
    match = re.match(r"^DEFWAVEFORM\s+([A-Za-z0-9_]+):$", line)
    if match is None:
        raise ValueError(f"Unrecognized DEFWAVEFORM header: {line}")
    return match.group(1)


def _parse_waveform_samples(lines: Sequence[str]) -> tuple[complex, ...]:
    payload = " ".join(lines).strip()
    if not payload:
        return ()
    tokens = [token.strip() for token in payload.split(",") if token.strip()]
    return tuple(_parse_complex_sample(token) for token in tokens)


def _parse_complex_sample(token: str) -> complex:
    return complex(token.replace("i", "j"))


def _parse_calibration_definition(
    lines: Sequence[str],
    start_index: int,
) -> tuple[_CalibrationDefinition, int]:
    header = lines[start_index].strip()
    match = re.match(r"^DEFCAL\s+([A-Z][A-Z0-9]*)(?:\((.*)\))?\s+(.+):$", header)
    if match is None:
        raise ValueError(f"Unrecognized DEFCAL header: {header}")
    name = match.group(1)
    raw_params = match.group(2)
    param_names = tuple(
        token.strip()
        for token in (raw_params.split(",") if raw_params else [])
        if token.strip()
    )
    qubits = tuple(int(token) for token in match.group(3).split())

    body_lines: list[str] = []
    index = start_index + 1
    while index < len(lines) and lines[index].startswith(" "):
        body_lines.append(lines[index].strip())
        index += 1

    return (
        _CalibrationDefinition(
            name=name,
            qubits=qubits,
            param_names=param_names,
            body_lines=tuple(body_lines),
        ),
        index,
    )


def _validate_invocation(
    invocation: _GateInvocation,
    model: _ProgramModel,
    calibration: CalibrationData,
    *,
    visited_blocks: set[tuple[str, tuple[int, ...], int]],
    missing_calibrations: list[str],
    missing_waveforms: list[str],
    missing_frames: list[str],
    unresolved_symbols: list[str],
    template_violations: list[str],
    substitutions: Mapping[str, str],
) -> None:
    definition = model.calibrations.get((invocation.name, invocation.qubits, len(invocation.params)))
    if definition is None:
        missing_calibrations.append(invocation.raw_line)
        return

    key = (definition.name, definition.qubits, len(definition.param_names))
    local_substitutions = dict(substitutions)
    local_substitutions.update(
        {
            name: value
            for name, value in zip(definition.param_names, invocation.params)
        }
    )
    if key not in visited_blocks:
        visited_blocks.add(key)
        _validate_template(definition, calibration, template_violations)

    for raw_line in definition.body_lines:
        line = _substitute_symbols(raw_line, local_substitutions)
        if "%" in line:
            unresolved_symbols.append(line)
        pulse = _parse_pulse_instruction(line)
        if pulse is not None:
            frame_key = (pulse[0], pulse[1])
            if frame_key not in model.frames:
                missing_frames.append(_frame_label(frame_key))
            if pulse[2] not in model.waveforms:
                missing_waveforms.append(pulse[2])
            continue
        nested = _parse_gate_invocation(line)
        if nested is not None and nested.name != "MEASURE":
            _validate_invocation(
                nested,
                model,
                calibration,
                visited_blocks=visited_blocks,
                missing_calibrations=missing_calibrations,
                missing_waveforms=missing_waveforms,
                missing_frames=missing_frames,
                unresolved_symbols=unresolved_symbols,
                template_violations=template_violations,
                substitutions=local_substitutions,
            )


def _validate_template(
    definition: _CalibrationDefinition,
    calibration: CalibrationData,
    template_violations: list[str],
) -> None:
    body = definition.body_lines
    if definition.name == "CNOT":
        if len(body) != 3 or not body[0].startswith("H ") or not body[1].startswith("CZ ") or not body[2].startswith("H "):
            template_violations.append(
                f"CNOT template on qubits {definition.qubits} does not match H-CZ-H."
            )
        return
    if definition.name == "SWAP":
        if len(body) != 3 or any(not line.startswith("CNOT ") for line in body):
            template_violations.append(
                f"SWAP template on qubits {definition.qubits} does not match the expected three-CNOT construction."
            )
        return
    if definition.name != "CZ":
        return

    edge = tuple(sorted(definition.qubits))
    fidelity = float(calibration.pair_metrics.get(edge, {}).get("fidelity", 1.0))
    pulse_count = sum(1 for line in body if line.startswith("PULSE "))
    x_count = sum(1 for line in body if re.match(r"^X\s+\d+$", line))
    if fidelity < 0.97:
        if pulse_count != 2 or x_count != 2:
            template_violations.append(
                f"Robust CZ template on edge {edge[0]}-{edge[1]} is missing the echoed pulse/refocusing structure."
            )
    elif pulse_count != 1:
        template_violations.append(
            f"Nominal CZ template on edge {edge[0]}-{edge[1]} should contain a single pulse."
        )


def _expand_invocation_to_events(
    invocation: _GateInvocation,
    model: _ProgramModel,
    *,
    start_ns: float,
    moment_index: int,
    frame_available: dict[tuple[tuple[int, ...], str], float],
    frame_scale: dict[tuple[tuple[int, ...], str], float],
    events: list[PulseEvent],
    missing_calibrations: list[str],
    missing_waveforms: list[str],
    missing_frames: list[str],
    unresolved_symbols: list[str],
    substitutions: Mapping[str, str],
) -> None:
    definition = model.calibrations.get((invocation.name, invocation.qubits, len(invocation.params)))
    if definition is None:
        missing_calibrations.append(invocation.raw_line)
        return

    local_substitutions = dict(substitutions)
    local_substitutions.update(
        {
            name: value
            for name, value in zip(definition.param_names, invocation.params)
        }
    )

    for raw_line in definition.body_lines:
        line = _substitute_symbols(raw_line, local_substitutions)
        if "%" in line:
            unresolved_symbols.append(line)

        pulse = _parse_pulse_instruction(line)
        if pulse is not None:
            targets, frame_name, waveform_name = pulse
            frame_key = (targets, frame_name)
            if frame_key not in model.frames:
                missing_frames.append(_frame_label(frame_key))
                continue
            waveform = model.waveforms.get(waveform_name)
            if waveform is None:
                missing_waveforms.append(waveform_name)
                continue
            current_start = max(start_ns, frame_available.get(frame_key, start_ns))
            amplitude = waveform.peak_amplitude * float(frame_scale.get(frame_key, 1.0))
            events.append(
                PulseEvent(
                    frame=_frame_label(frame_key),
                    waveform=waveform.name,
                    qubits=targets,
                    source_gate=f"{invocation.name}{invocation.qubits}",
                    moment_index=int(moment_index),
                    start_ns=float(current_start),
                    duration_ns=float(waveform.duration_ns),
                    amplitude=float(amplitude),
                )
            )
            frame_available[frame_key] = current_start + waveform.duration_ns
            continue

        scale = _parse_set_scale_instruction(line)
        if scale is not None:
            targets, frame_name, expression = scale
            frame_key = (targets, frame_name)
            if frame_key in model.frames:
                frame_scale[frame_key] = _evaluate_numeric_expression(expression)
            continue

        phase = _parse_shift_phase_instruction(line)
        if phase is not None:
            targets, frame_name, _ = phase
            frame_key = (targets, frame_name)
            if frame_key in model.frames:
                frame_available[frame_key] = max(frame_available.get(frame_key, start_ns), start_ns)
            continue

        fence_targets = _parse_fence_instruction(line)
        if fence_targets is not None:
            related = _related_frames(fence_targets, model)
            if related:
                synchronized = max(frame_available.get(frame_key, start_ns) for frame_key in related)
                for frame_key in related:
                    frame_available[frame_key] = synchronized
            continue

        nested = _parse_gate_invocation(line)
        if nested is not None and nested.name != "MEASURE":
            _expand_invocation_to_events(
                nested,
                model,
                start_ns=start_ns,
                moment_index=moment_index,
                frame_available=frame_available,
                frame_scale=frame_scale,
                events=events,
                missing_calibrations=missing_calibrations,
                missing_waveforms=missing_waveforms,
                missing_frames=missing_frames,
                unresolved_symbols=unresolved_symbols,
                substitutions=local_substitutions,
            )


def _parse_pulse_instruction(line: str) -> tuple[tuple[int, ...], str, str] | None:
    match = re.match(r'^PULSE\s+(.+?)\s+"([^"]+)"\s+([A-Za-z0-9_]+)$', line)
    if match is None:
        return None
    targets = tuple(int(token) for token in match.group(1).split())
    return targets, match.group(2), match.group(3)


def _parse_set_scale_instruction(line: str) -> tuple[tuple[int, ...], str, str] | None:
    match = re.match(r'^SET-SCALE\s+(.+?)\s+"([^"]+)"\s+(.+)$', line)
    if match is None:
        return None
    targets = tuple(int(token) for token in match.group(1).split())
    return targets, match.group(2), match.group(3)


def _parse_shift_phase_instruction(line: str) -> tuple[tuple[int, ...], str, str] | None:
    match = re.match(r'^SHIFT-PHASE\s+(.+?)\s+"([^"]+)"\s+(.+)$', line)
    if match is None:
        return None
    targets = tuple(int(token) for token in match.group(1).split())
    return targets, match.group(2), match.group(3)


def _parse_fence_instruction(line: str) -> tuple[int, ...] | None:
    match = re.match(r"^FENCE\s+([\d\s]+)$", line)
    if match is None:
        return None
    return tuple(int(token) for token in match.group(1).split())


def _parse_gate_invocation(line: str) -> _GateInvocation | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or stripped.startswith("DECLARE"):
        return None
    if stripped.startswith(("PULSE", "SET-SCALE", "SHIFT-PHASE", "DELAY", "FENCE", "DEFCAL", "DEFFRAME", "DEFWAVEFORM")):
        return None

    measure = re.match(r"^MEASURE\s+(\d+)\s+ro\[(\d+)\]$", stripped)
    if measure is not None:
        return _GateInvocation(
            name="MEASURE",
            qubits=(int(measure.group(1)),),
            params=(measure.group(2),),
            raw_line=stripped,
        )

    angle = re.match(r"^([A-Z][A-Z0-9]*)\((.+)\)\s+([\d\s]+)$", stripped)
    if angle is not None:
        return _GateInvocation(
            name=angle.group(1),
            qubits=tuple(int(token) for token in angle.group(3).split()),
            params=(angle.group(2).strip(),),
            raw_line=stripped,
        )

    plain = re.match(r"^([A-Z][A-Z0-9]*)\s+([\d\s]+)$", stripped)
    if plain is not None and plain.group(1) in (_SINGLE_QUBIT_GATES | _TWO_QUBIT_GATES):
        return _GateInvocation(
            name=plain.group(1),
            qubits=tuple(int(token) for token in plain.group(2).split()),
            params=(),
            raw_line=stripped,
        )

    return None


def _substitute_symbols(line: str, substitutions: Mapping[str, str]) -> str:
    resolved = line
    for name, value in substitutions.items():
        resolved = resolved.replace(name, f"({value})")
    return resolved


def _evaluate_numeric_expression(expression: str) -> float:
    parsed = ast.parse(expression, mode="eval")
    return float(_SafeEvaluator().visit(parsed))


def _frame_label(frame_key: tuple[tuple[int, ...], str]) -> str:
    targets, frame_name = frame_key
    return f'{" ".join(str(target) for target in targets)} "{frame_name}"'


def _frames_for_invocation(
    invocation: _GateInvocation,
    model: _ProgramModel,
) -> list[tuple[tuple[int, ...], str]]:
    if invocation.name in _SINGLE_QUBIT_GATES:
        frame_key = ((int(invocation.qubits[0]),), "xy")
        return [frame_key] if frame_key in model.frames else []
    if invocation.name in {"CNOT", "CZ", "SWAP"}:
        edge = tuple(int(qubit) for qubit in invocation.qubits)
        candidates = [
            (edge, "cz"),
            (tuple(sorted(edge)), "cz"),
        ]
        return [frame_key for frame_key in candidates if frame_key in model.frames]
    return []


def _related_frames(
    qubits: Sequence[int],
    model: _ProgramModel,
) -> list[tuple[tuple[int, ...], str]]:
    qubit_set = set(int(qubit) for qubit in qubits)
    related: list[tuple[tuple[int, ...], str]] = []
    for frame_key in model.frames:
        targets, _ = frame_key
        if qubit_set.intersection(targets):
            related.append(frame_key)
    return related


def _max_simultaneous_pulse_events(events: Sequence[PulseEvent]) -> int:
    markers: list[tuple[float, int]] = []
    for event in events:
        markers.append((float(event.start_ns), 1))
        markers.append((float(event.start_ns + event.duration_ns), -1))
    markers.sort(key=lambda item: (item[0], -item[1]))
    active = 0
    maximum = 0
    for _, delta in markers:
        active += delta
        maximum = max(maximum, active)
    return int(maximum)


def _alignment_violations(
    result: CompilationResult,
    *,
    tolerance_ns: float,
    sensitive_zz_mhz: float,
) -> list[str]:
    violations: list[str] = []
    base_cycle_ns = float(result.calibration.default_1q_duration_ns)

    for left, right in result.calibration.topology.edges():
        pair_metric = result.calibration.pair_metrics.get((int(left), int(right)), {})
        if float(pair_metric.get("zz_mhz", 0.0)) <= float(sensitive_zz_mhz):
            continue

        left_starts: list[float] = []
        right_starts: list[float] = []
        for moment_index, moment in enumerate(result.moments):
            start_ns = float(moment_index * base_cycle_ns)
            for op in moment:
                if op.source != "dd":
                    continue
                if op.qubits == (int(left),):
                    left_starts.append(start_ns)
                elif op.qubits == (int(right),):
                    right_starts.append(start_ns)

        if len(left_starts) != len(right_starts):
            violations.append(
                (
                    f"Sensitive pair {int(left)}-{int(right)} has mismatched DD counts "
                    f"({len(left_starts)} vs {len(right_starts)})."
                )
            )
            continue

        for index, (left_start, right_start) in enumerate(zip(left_starts, right_starts)):
            if abs(left_start - right_start) > tolerance_ns:
                violations.append(
                    (
                        f"Sensitive pair {int(left)}-{int(right)} DD pulse {index} starts at "
                        f"{left_start:.6f} ns vs {right_start:.6f} ns."
                    )
                )

    return violations


def _invocation_from_physical_operation(op: PhysicalOperation) -> _GateInvocation:
    raw_line = _compiler._emit_instruction(op)[0]
    invocation = _parse_gate_invocation(raw_line)
    if invocation is None:
        raise ValueError(f"Could not reconstruct an invocation from '{raw_line}'.")
    return invocation


def _simulate_density_matrix(
    result: CompilationResult,
    *,
    include_zz: bool,
    include_decoherence: bool = True,
) -> np.ndarray:
    simulated_qubits = sorted({int(qubit) for op in result.operations for qubit in op.qubits})
    if not simulated_qubits:
        return np.array([[1.0 + 0.0j]], dtype=complex)
    if len(simulated_qubits) > 4:
        raise ValueError("The effective analog simulator only supports up to four active qubits.")

    local_index = {qubit: index for index, qubit in enumerate(simulated_qubits)}
    dimension = 1 << len(simulated_qubits)
    rho = np.zeros((dimension, dimension), dtype=complex)
    rho[0, 0] = 1.0
    toggles = {qubit: 1 for qubit in simulated_qubits}

    for moment in result.moments:
        duration_ns = max((_compiler._operation_duration_ns(op, result.calibration) for op in moment), default=0.0)
        for op in moment:
            if op.name == "measure":
                continue
            operator = _gate_unitary(op)
            qubits = tuple(local_index[int(qubit)] for qubit in op.qubits)
            rho = _apply_unitary_channel(rho, operator, qubits, len(simulated_qubits))
            if op.name in {"x", "y"}:
                toggles[int(op.qubits[0])] *= -1

        if include_zz and duration_ns > 0.0:
            for left, right in result.calibration.topology.edges():
                left = int(left)
                right = int(right)
                if left not in local_index or right not in local_index:
                    continue
                zz_mhz = float(
                    result.calibration.pair_metrics.get((left, right), {}).get(
                        "zz_mhz",
                        result.calibration.pair_metrics.get((right, left), {}).get("zz_mhz", 0.0),
                    )
                )
                if abs(zz_mhz) <= EPS:
                    continue
                theta = 2.0 * math.pi * zz_mhz * float(duration_ns) * 1.0e-3
                theta *= float(toggles[left] * toggles[right])
                zz_unitary = np.diag(
                    np.exp(
                        -0.5j * theta * np.array([1.0, -1.0, -1.0, 1.0], dtype=float)
                    )
                )
                rho = _apply_unitary_channel(
                    rho,
                    zz_unitary,
                    (local_index[left], local_index[right]),
                    len(simulated_qubits),
                )

        if include_decoherence and duration_ns > 0.0:
            for physical in simulated_qubits:
                metrics = result.calibration.qubit_metrics.get(int(physical), {})
                t1_us = float(metrics.get("t1", 20.0))
                t2_us = float(metrics.get("t2", 20.0))
                rho = _apply_amplitude_damping(
                    rho,
                    gamma=_amplitude_damping_probability(duration_ns, t1_us),
                    qubit=local_index[int(physical)],
                    qubit_count=len(simulated_qubits),
                )
                rho = _apply_phase_damping(
                    rho,
                    probability=_phase_damping_probability(duration_ns, t1_us, t2_us),
                    qubit=local_index[int(physical)],
                    qubit_count=len(simulated_qubits),
                )

    return rho


def _gate_unitary(op: PhysicalOperation) -> np.ndarray:
    if op.name == "h":
        return (1.0 / math.sqrt(2.0)) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex)
    if op.name == "x":
        return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    if op.name == "y":
        return np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    if op.name == "z":
        return np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    if op.name == "rx":
        angle = float(op.params[0]) / 2.0
        return np.array(
            [
                [math.cos(angle), -1.0j * math.sin(angle)],
                [-1.0j * math.sin(angle), math.cos(angle)],
            ],
            dtype=complex,
        )
    if op.name == "ry":
        angle = float(op.params[0]) / 2.0
        return np.array(
            [
                [math.cos(angle), -math.sin(angle)],
                [math.sin(angle), math.cos(angle)],
            ],
            dtype=complex,
        )
    if op.name == "rz":
        angle = float(op.params[0]) / 2.0
        return np.array(
            [
                [np.exp(-1.0j * angle), 0.0],
                [0.0, np.exp(1.0j * angle)],
            ],
            dtype=complex,
        )
    if op.name == "cx":
        return np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=complex,
        )
    if op.name == "cz":
        return np.diag([1.0, 1.0, 1.0, -1.0]).astype(complex)
    if op.name == "swap":
        return np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=complex,
        )
    raise ValueError(f"Unsupported analog simulation gate '{op.name}'.")


def _apply_unitary_channel(
    rho: np.ndarray,
    operator: np.ndarray,
    qubits: tuple[int, ...],
    qubit_count: int,
) -> np.ndarray:
    embedded = _embed_operator(operator, qubits, qubit_count)
    return embedded @ rho @ embedded.conj().T


def _embed_operator(
    operator: np.ndarray,
    qubits: tuple[int, ...],
    qubit_count: int,
) -> np.ndarray:
    qubits = tuple(int(qubit) for qubit in qubits)
    dimension = 1 << qubit_count
    embedded = np.zeros((dimension, dimension), dtype=complex)
    subdimension = 1 << len(qubits)

    for source in range(dimension):
        source_bits = _index_bits(source, qubit_count)
        sub_source = _bits_to_index(source_bits[index] for index in qubits)
        for sub_target in range(subdimension):
            amplitude = operator[sub_target, sub_source]
            if abs(amplitude) <= EPS:
                continue
            target_bits = list(source_bits)
            sub_target_bits = _index_bits(sub_target, len(qubits))
            for offset, qubit in enumerate(qubits):
                target_bits[qubit] = sub_target_bits[offset]
            target = _bits_to_index(target_bits)
            embedded[target, source] = amplitude

    return embedded


def _index_bits(index: int, width: int) -> list[int]:
    return [(int(index) >> shift) & 1 for shift in range(width - 1, -1, -1)]


def _bits_to_index(bits: Sequence[int]) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return int(value)


def _apply_amplitude_damping(
    rho: np.ndarray,
    *,
    gamma: float,
    qubit: int,
    qubit_count: int,
) -> np.ndarray:
    gamma = max(0.0, min(1.0, float(gamma)))
    if gamma <= 0.0:
        return rho
    k0 = np.array([[1.0, 0.0], [0.0, math.sqrt(1.0 - gamma)]], dtype=complex)
    k1 = np.array([[0.0, math.sqrt(gamma)], [0.0, 0.0]], dtype=complex)
    return _apply_kraus_channel(rho, (k0, k1), qubit=qubit, qubit_count=qubit_count)


def _apply_phase_damping(
    rho: np.ndarray,
    *,
    probability: float,
    qubit: int,
    qubit_count: int,
) -> np.ndarray:
    probability = max(0.0, min(0.5, float(probability)))
    if probability <= 0.0:
        return rho
    identity = np.eye(2, dtype=complex)
    z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    a = math.sqrt(max(0.0, 1.0 - probability))
    b = math.sqrt(max(0.0, probability))
    return _apply_kraus_channel(
        rho,
        (a * identity, b * z),
        qubit=qubit,
        qubit_count=qubit_count,
    )


def _apply_kraus_channel(
    rho: np.ndarray,
    kraus: Sequence[np.ndarray],
    *,
    qubit: int,
    qubit_count: int,
) -> np.ndarray:
    updated = np.zeros_like(rho)
    for operator in kraus:
        embedded = _embed_operator(operator, (qubit,), qubit_count)
        updated += embedded @ rho @ embedded.conj().T
    return updated


def _amplitude_damping_probability(duration_ns: float, t1_us: float) -> float:
    t1_ns = max(float(t1_us) * 1000.0, EPS)
    return 1.0 - math.exp(-float(duration_ns) / t1_ns)


def _phase_damping_probability(duration_ns: float, t1_us: float, t2_us: float) -> float:
    t1_inv = 1.0 / max(float(t1_us), EPS)
    t2_inv = 1.0 / max(float(t2_us), EPS)
    pure_inv = max(0.0, t2_inv - 0.5 * t1_inv)
    if pure_inv <= EPS:
        return 0.0
    tphi_ns = 1000.0 / pure_inv
    coherence = math.exp(-float(duration_ns) / max(tphi_ns, EPS))
    return max(0.0, min(0.5, 0.5 * (1.0 - coherence)))


def _measurement_probabilities(
    rho: np.ndarray,
    result: CompilationResult,
) -> dict[str, float]:
    measured = [
        (int(op.clbits[0]), int(op.qubits[0]))
        for op in result.operations
        if op.name == "measure" and op.clbits
    ]
    active_qubits = sorted({int(qubit) for op in result.operations for qubit in op.qubits})
    local_index = {qubit: index for index, qubit in enumerate(active_qubits)}

    if not measured:
        ordered_qubits = active_qubits
    else:
        ordered_qubits = [physical for _, physical in sorted(measured)]

    diagonal = np.real_if_close(np.diag(rho))
    probabilities: dict[str, float] = {}
    for state_index, weight in enumerate(diagonal):
        bits = _index_bits(state_index, len(active_qubits))
        bitstring = "".join(str(bits[local_index[physical]]) for physical in ordered_qubits)
        probabilities[bitstring] = probabilities.get(bitstring, 0.0) + float(weight)

    total = sum(probabilities.values())
    if total > 0.0:
        probabilities = {bitstring: value / total for bitstring, value in probabilities.items()}
    return probabilities


def _apply_readout_model(
    probabilities: Mapping[str, float],
    result: CompilationResult,
) -> dict[str, float]:
    if not probabilities:
        return {}

    measured = [
        (int(op.clbits[0]), int(op.qubits[0]))
        for op in result.operations
        if op.name == "measure" and op.clbits
    ]
    ordered_qubits = [physical for _, physical in sorted(measured)]
    if not ordered_qubits:
        ordered_qubits = sorted({int(qubit) for op in result.operations for qubit in op.qubits})

    output: dict[str, float] = {}
    for observed in sorted(probabilities):
        accumulated = 0.0
        for actual, probability in probabilities.items():
            confusion = 1.0
            for index, physical in enumerate(ordered_qubits):
                fidelity = float(
                    result.calibration.qubit_metrics.get(int(physical), {}).get("readout_fidelity", 0.95)
                )
                flip = max(0.0, min(0.5 - EPS, 1.0 - fidelity))
                confusion *= (1.0 - flip) if observed[index] == actual[index] else flip
            accumulated += confusion * float(probability)
        output[observed] = accumulated

    total = sum(output.values())
    if total > 0.0:
        output = {bitstring: value / total for bitstring, value in output.items()}
    return output


def _parity_expectation(probabilities: Mapping[str, float]) -> float:
    expectation = 0.0
    for bitstring, probability in probabilities.items():
        expectation += ((-1.0) ** sum(int(bit) for bit in bitstring)) * float(probability)
    return float(expectation)
