from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


DEFAULT_ANKAA3_SPECS_PATH = "Ankaa-3_device_specs.csv"


@dataclass(frozen=True)
class DeviceErrorModel:
    name: str
    t1_microseconds: Mapping[int, float]
    t2_microseconds: Mapping[int, float]
    one_qubit_fidelity: Mapping[int, float]
    readout_fidelity: Mapping[int, float]
    two_qubit_fidelity: Mapping[tuple[int, int], float]
    avg_t1_microseconds: float
    avg_t2_microseconds: float
    avg_one_qubit_fidelity: float
    avg_readout_fidelity: float
    avg_two_qubit_fidelity: float


@dataclass(frozen=True)
class BuiltNoiseModel:
    name: str
    noise_model: Any
    physical_layout: tuple[int, ...]
    single_qubit_gate_names: tuple[str, ...]
    two_qubit_gate_names: tuple[str, ...]
    one_qubit_gate_time_ns: float
    two_qubit_gate_time_ns: float
    measure_time_ns: float


def _pair_key(left: int, right: int) -> tuple[int, int]:
    a = int(left)
    b = int(right)
    return (a, b) if a <= b else (b, a)


def _parse_float(value: str | None) -> float | None:
    raw = (value or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


@lru_cache(maxsize=8)
def load_ankaa3_error_model(path_str: str = DEFAULT_ANKAA3_SPECS_PATH) -> DeviceErrorModel:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Device spec file was not found: {path}")

    t1_microseconds: dict[int, float] = {}
    t2_microseconds: dict[int, float] = {}
    one_qubit_fidelity: dict[int, float] = {}
    readout_fidelity: dict[int, float] = {}
    two_qubit_fidelity: dict[tuple[int, int], float] = {}

    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            qubit_text = (row.get("Qubit") or "").strip()
            pair_text = (row.get("Pair") or "").strip()

            if qubit_text:
                try:
                    qubit = int(qubit_text)
                except ValueError:
                    continue
                t1 = _parse_float(row.get("T1 (\u00c2\u00b5s)") or row.get("T1 (Âµs)") or row.get("T1 (\u00b5s)"))
                t2 = _parse_float(row.get("T2 (\u00c2\u00b5s)") or row.get("T2 (Âµs)") or row.get("T2 (\u00b5s)"))
                f1q = _parse_float(row.get("f1QRB"))
                fro = _parse_float(row.get("fRO"))
                if t1 is not None:
                    t1_microseconds[qubit] = max(0.0, t1)
                if t2 is not None:
                    t2_microseconds[qubit] = max(0.0, t2)
                if f1q is not None:
                    one_qubit_fidelity[qubit] = max(0.0, min(1.0, f1q))
                if fro is not None:
                    readout_fidelity[qubit] = max(0.0, min(1.0, fro))
                continue

            if pair_text and "-" in pair_text:
                parts = pair_text.split("-", maxsplit=1)
                try:
                    left = int(parts[0])
                    right = int(parts[1])
                except ValueError:
                    continue
                fiswap = _parse_float(row.get("fISWAP"))
                if fiswap is not None:
                    two_qubit_fidelity[_pair_key(left, right)] = max(0.0, min(1.0, fiswap))

    if not t1_microseconds or not t2_microseconds or not one_qubit_fidelity or not readout_fidelity or not two_qubit_fidelity:
        raise ValueError(f"Device spec file is incomplete: {path}")

    return DeviceErrorModel(
        name="ankaa3",
        t1_microseconds=t1_microseconds,
        t2_microseconds=t2_microseconds,
        one_qubit_fidelity=one_qubit_fidelity,
        readout_fidelity=readout_fidelity,
        two_qubit_fidelity=two_qubit_fidelity,
        avg_t1_microseconds=float(np.mean(list(t1_microseconds.values()))),
        avg_t2_microseconds=float(np.mean(list(t2_microseconds.values()))),
        avg_one_qubit_fidelity=float(np.mean(list(one_qubit_fidelity.values()))),
        avg_readout_fidelity=float(np.mean(list(readout_fidelity.values()))),
        avg_two_qubit_fidelity=float(np.mean(list(two_qubit_fidelity.values()))),
    )


def _candidate_layouts(
    model: DeviceErrorModel,
    num_qubits: int,
    layout_seed: int,
) -> list[list[int]]:
    target_size = max(1, int(num_qubits))
    hardware_graph: dict[int, list[tuple[int, float]]] = {}
    for (left, right), fidelity in model.two_qubit_fidelity.items():
        hardware_graph.setdefault(left, []).append((right, float(fidelity)))
        hardware_graph.setdefault(right, []).append((left, float(fidelity)))

    for node in hardware_graph:
        hardware_graph[node].sort(key=lambda item: (-item[1], item[0]))

    candidates: list[list[int]] = []
    ordered_starts = sorted(
        model.one_qubit_fidelity,
        key=lambda node: (
            -(model.one_qubit_fidelity.get(node, model.avg_one_qubit_fidelity) + model.readout_fidelity.get(node, model.avg_readout_fidelity)),
            node,
        ),
    )

    offset = max(0, int(layout_seed)) % max(1, len(ordered_starts))
    rotated_starts = ordered_starts[offset:] + ordered_starts[:offset]

    for start in rotated_starts[: min(16, len(rotated_starts))]:
        path = [int(start)]
        used = {int(start)}
        while len(path) < target_size:
            current = path[-1]
            neighbors = [(node, fidelity) for node, fidelity in hardware_graph.get(current, ()) if node not in used]
            if not neighbors:
                break
            next_node = max(
                neighbors,
                key=lambda item: (
                    item[1]
                    * model.one_qubit_fidelity.get(item[0], model.avg_one_qubit_fidelity)
                    * model.readout_fidelity.get(item[0], model.avg_readout_fidelity),
                    -item[0],
                ),
            )[0]
            path.append(int(next_node))
            used.add(int(next_node))
        if len(path) == target_size:
            candidates.append(path)

    if candidates:
        return candidates

    fallback = sorted(
        model.one_qubit_fidelity,
        key=lambda node: (
            -(model.one_qubit_fidelity.get(node, model.avg_one_qubit_fidelity) + model.readout_fidelity.get(node, model.avg_readout_fidelity)),
            node,
        ),
    )[:target_size]
    return [fallback]


def select_ankaa3_layout(
    num_qubits: int,
    *,
    specs_path: str = DEFAULT_ANKAA3_SPECS_PATH,
    layout_seed: int = 0,
) -> tuple[int, ...]:
    model = load_ankaa3_error_model(specs_path)
    layouts = _candidate_layouts(model, num_qubits=num_qubits, layout_seed=layout_seed)
    if not layouts:
        return ()
    return tuple(int(node) for node in layouts[0])


def _clamp_t2_seconds(t1_seconds: float, t2_seconds: float) -> float:
    if t1_seconds <= 0.0:
        return max(1e-9, t2_seconds)
    return max(1e-9, min(t2_seconds, 2.0 * t1_seconds))


def build_ankaa3_noise_model(
    num_qubits: int,
    *,
    specs_path: str = DEFAULT_ANKAA3_SPECS_PATH,
    layout_seed: int = 0,
    single_qubit_gate_names: Sequence[str] = ("h", "rx", "ry", "rz", "x", "sx"),
    two_qubit_gate_names: Sequence[str] = ("rzz", "cx", "iswap", "ecr"),
    one_qubit_gate_time_ns: float = 40.0,
    two_qubit_gate_time_ns: float = 220.0,
    measure_time_ns: float = 1200.0,
) -> BuiltNoiseModel:
    try:
        from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error, thermal_relaxation_error
    except ImportError as exc:
        raise ImportError(
            "qiskit_aer is required to build a concrete Ankaa-3 noise model."
        ) from exc

    logical_qubits = max(0, int(num_qubits))
    if logical_qubits < 1:
        return BuiltNoiseModel(
            name="ankaa3",
            noise_model=NoiseModel(),
            physical_layout=(),
            single_qubit_gate_names=tuple(single_qubit_gate_names),
            two_qubit_gate_names=tuple(two_qubit_gate_names),
            one_qubit_gate_time_ns=float(one_qubit_gate_time_ns),
            two_qubit_gate_time_ns=float(two_qubit_gate_time_ns),
            measure_time_ns=float(measure_time_ns),
        )

    model = load_ankaa3_error_model(specs_path)
    layout = select_ankaa3_layout(logical_qubits, specs_path=specs_path, layout_seed=layout_seed)
    if len(layout) != logical_qubits:
        raise ValueError(
            f"Unable to place {logical_qubits} logical qubits on Ankaa-3 using the current layout heuristic."
        )

    noise_model = NoiseModel()
    one_qubit_time_s = max(0.0, float(one_qubit_gate_time_ns)) * 1e-9
    two_qubit_time_s = max(0.0, float(two_qubit_gate_time_ns)) * 1e-9
    measure_time_s = max(0.0, float(measure_time_ns)) * 1e-9

    for logical_qubit, physical_qubit in enumerate(layout):
        t1_seconds = max(1e-9, float(model.t1_microseconds.get(physical_qubit, model.avg_t1_microseconds)) * 1e-6)
        raw_t2_seconds = max(1e-9, float(model.t2_microseconds.get(physical_qubit, model.avg_t2_microseconds)) * 1e-6)
        t2_seconds = _clamp_t2_seconds(t1_seconds, raw_t2_seconds)

        one_qubit_error_rate = max(0.0, min(1.0, 1.0 - float(model.one_qubit_fidelity.get(physical_qubit, model.avg_one_qubit_fidelity))))
        readout_error_rate = max(0.0, min(1.0, 1.0 - float(model.readout_fidelity.get(physical_qubit, model.avg_readout_fidelity))))

        gate_error = depolarizing_error(one_qubit_error_rate, 1).compose(
            thermal_relaxation_error(t1_seconds, t2_seconds, one_qubit_time_s)
        )
        for gate_name in dict.fromkeys(str(name) for name in single_qubit_gate_names):
            noise_model.add_quantum_error(gate_error, gate_name, [logical_qubit])

        measurement_error = ReadoutError(
            [
                [1.0 - readout_error_rate, readout_error_rate],
                [readout_error_rate, 1.0 - readout_error_rate],
            ]
        )
        noise_model.add_readout_error(measurement_error, [logical_qubit])

        if measure_time_s > 0.0:
            measure_relaxation = thermal_relaxation_error(t1_seconds, t2_seconds, measure_time_s)
            noise_model.add_quantum_error(measure_relaxation, "measure", [logical_qubit])

    for logical_left in range(logical_qubits - 1):
        logical_right = logical_left + 1
        physical_left = layout[logical_left]
        physical_right = layout[logical_right]
        pair_fidelity = float(
            model.two_qubit_fidelity.get(
                _pair_key(physical_left, physical_right),
                model.avg_two_qubit_fidelity,
            )
        )
        two_qubit_error_rate = max(0.0, min(1.0, 1.0 - pair_fidelity))

        left_t1 = max(1e-9, float(model.t1_microseconds.get(physical_left, model.avg_t1_microseconds)) * 1e-6)
        left_t2 = _clamp_t2_seconds(
            left_t1,
            max(1e-9, float(model.t2_microseconds.get(physical_left, model.avg_t2_microseconds)) * 1e-6),
        )
        right_t1 = max(1e-9, float(model.t1_microseconds.get(physical_right, model.avg_t1_microseconds)) * 1e-6)
        right_t2 = _clamp_t2_seconds(
            right_t1,
            max(1e-9, float(model.t2_microseconds.get(physical_right, model.avg_t2_microseconds)) * 1e-6),
        )

        thermal_pair = thermal_relaxation_error(left_t1, left_t2, two_qubit_time_s).expand(
            thermal_relaxation_error(right_t1, right_t2, two_qubit_time_s)
        )
        two_qubit_error = depolarizing_error(two_qubit_error_rate, 2).compose(thermal_pair)

        for gate_name in dict.fromkeys(str(name) for name in two_qubit_gate_names):
            noise_model.add_quantum_error(two_qubit_error, gate_name, [logical_left, logical_right])

    return BuiltNoiseModel(
        name="ankaa3",
        noise_model=noise_model,
        physical_layout=layout,
        single_qubit_gate_names=tuple(dict.fromkeys(str(name) for name in single_qubit_gate_names)),
        two_qubit_gate_names=tuple(dict.fromkeys(str(name) for name in two_qubit_gate_names)),
        one_qubit_gate_time_ns=float(one_qubit_gate_time_ns),
        two_qubit_gate_time_ns=float(two_qubit_gate_time_ns),
        measure_time_ns=float(measure_time_ns),
    )


@lru_cache(maxsize=32)
def _get_cached_ankaa3_density_simulator(
    num_qubits: int,
    specs_path: str,
    layout_seed: int,
) -> tuple[BuiltNoiseModel, Any]:
    from qiskit_aer import AerSimulator

    built = build_ankaa3_noise_model(
        num_qubits,
        specs_path=specs_path,
        layout_seed=layout_seed,
    )
    simulator = AerSimulator(method="density_matrix", noise_model=built.noise_model)
    return built, simulator


def estimate_ankaa3_circuit_fidelity(
    num_qubits: int,
    *,
    one_qubit_gate_count: int,
    two_qubit_gate_count: int,
    measurement_qubits: int,
    specs_path: str = DEFAULT_ANKAA3_SPECS_PATH,
    layout_seed: int = 0,
) -> float:
    model = load_ankaa3_error_model(specs_path)
    layouts = _candidate_layouts(model, num_qubits=num_qubits, layout_seed=layout_seed)
    best_fidelity = 0.0

    for layout in layouts:
        if not layout:
            continue
        one_qubit_fidelities = [model.one_qubit_fidelity.get(node, model.avg_one_qubit_fidelity) for node in layout]
        readout_fidelities = [model.readout_fidelity.get(node, model.avg_readout_fidelity) for node in layout]
        chain_edge_fidelities = []
        for index in range(len(layout) - 1):
            chain_edge_fidelities.append(
                model.two_qubit_fidelity.get(
                    _pair_key(layout[index], layout[index + 1]),
                    model.avg_two_qubit_fidelity,
                )
            )

        avg_one = float(np.mean(one_qubit_fidelities)) if one_qubit_fidelities else model.avg_one_qubit_fidelity
        avg_readout = float(np.mean(readout_fidelities)) if readout_fidelities else model.avg_readout_fidelity
        avg_two = float(np.mean(chain_edge_fidelities)) if chain_edge_fidelities else model.avg_two_qubit_fidelity

        fidelity = (
            avg_one ** max(0, int(one_qubit_gate_count))
            * avg_two ** max(0, int(two_qubit_gate_count))
            * avg_readout ** max(0, min(int(measurement_qubits), len(layout)))
        )
        best_fidelity = max(best_fidelity, float(fidelity))

    return max(0.0, min(1.0, best_fidelity))


def simulate_circuit_probabilities_with_error_model(
    circuit: Any,
    *,
    num_qubits: int,
    one_qubit_gate_count: int,
    two_qubit_gate_count: int,
    measurement_qubits: int,
    error_model: str,
    specs_path: str = DEFAULT_ANKAA3_SPECS_PATH,
    layout_seed: int = 0,
    shots: int = 0,
) -> tuple[np.ndarray, float]:
    from qiskit.quantum_info import Statevector

    normalized_model = error_model.strip().lower()
    if normalized_model == "ideal":
        state = Statevector.from_instruction(circuit)
        probabilities = np.abs(state.data) ** 2
        return np.asarray(probabilities, dtype=float), 1.0

    if normalized_model == "ankaa3":
        state = Statevector.from_instruction(circuit)
        probabilities = np.abs(state.data) ** 2
        return apply_modular_error_model_to_probabilities(
            np.asarray(probabilities, dtype=float),
            num_qubits=num_qubits,
            one_qubit_gate_count=one_qubit_gate_count,
            two_qubit_gate_count=two_qubit_gate_count,
            measurement_qubits=measurement_qubits,
            error_model=error_model,
            specs_path=specs_path,
            layout_seed=layout_seed,
        )

    if normalized_model != "ankaa3_hardware":
        raise ValueError(f"Unknown error model: {error_model}")

    _, simulator = _get_cached_ankaa3_density_simulator(
        max(0, int(num_qubits)),
        str(specs_path),
        int(layout_seed),
    )
    shot_count = max(0, int(shots))
    run_circuit = circuit.copy()
    if shot_count > 0:
        run_circuit.measure_all()
        result = simulator.run(run_circuit, shots=shot_count).result()
        counts = result.get_counts(0)
        basis_size = max(1, 1 << max(0, int(num_qubits)))
        probabilities = np.zeros(basis_size, dtype=float)
        total = 0.0
        for bitstring, count in counts.items():
            key = str(bitstring).replace(" ", "")
            if not key:
                continue
            probabilities[int(key, 2)] += float(count)
            total += float(count)
    else:
        run_circuit.save_probabilities()
        result = simulator.run(run_circuit).result()
        probabilities = np.asarray(result.data(0)["probabilities"], dtype=float)
        total = float(np.sum(probabilities))

    if not math.isfinite(total) or total <= 0.0:
        basis_size = max(1, len(probabilities))
        probabilities = np.full(basis_size, 1.0 / basis_size, dtype=float)
    else:
        probabilities = probabilities / total

    fidelity = estimate_ankaa3_circuit_fidelity(
        num_qubits,
        one_qubit_gate_count=one_qubit_gate_count,
        two_qubit_gate_count=two_qubit_gate_count,
        measurement_qubits=measurement_qubits,
        specs_path=specs_path,
        layout_seed=layout_seed,
    )
    return probabilities, fidelity


def apply_modular_error_model_to_probabilities(
    probabilities: np.ndarray,
    *,
    num_qubits: int,
    one_qubit_gate_count: int,
    two_qubit_gate_count: int,
    measurement_qubits: int,
    error_model: str,
    specs_path: str = DEFAULT_ANKAA3_SPECS_PATH,
    layout_seed: int = 0,
) -> tuple[np.ndarray, float]:
    normalized_model = error_model.strip().lower()
    if normalized_model == "ideal":
        return probabilities, 1.0
    if normalized_model != "ankaa3":
        raise ValueError(f"Unknown error model: {error_model}")

    fidelity = estimate_ankaa3_circuit_fidelity(
        num_qubits,
        one_qubit_gate_count=one_qubit_gate_count,
        two_qubit_gate_count=two_qubit_gate_count,
        measurement_qubits=measurement_qubits,
        specs_path=specs_path,
        layout_seed=layout_seed,
    )
    basis_size = max(1, len(probabilities))
    uniform = np.full(basis_size, 1.0 / basis_size, dtype=float)
    noisy = fidelity * np.asarray(probabilities, dtype=float) + (1.0 - fidelity) * uniform
    total = float(np.sum(noisy))
    if not math.isfinite(total) or total <= 0.0:
        return uniform, fidelity
    return noisy / total, fidelity
