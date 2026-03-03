"""Credit-based resource accounting for modular benchmark runs.

This module implements a normalized "work budget" model used by the
benchmarking wrappers. The goal is not to predict wall-clock time. Instead, it
assigns comparable credit counts to quantum and classical work so runs can be
compared by "solution quality achieved after X credits".

The charging rules used here are intentionally simple:

- Quantum objective/statevector evaluations charge `credit_cost` credits
  (default: 1) and also record proxy size metrics such as simulated amplitudes
  and amplitude-level operation counts.
- Quantum decode steps charge `credit_cost` credits (default: 1) and record how
  many amplitudes were scanned while extracting a bitstring.
- Classical exact block solves charge by the number of states enumerated, plus
  any extra tie-break states that had to be evaluated.
- Classical heuristic passes charge by `node_visits`, which makes local-search
  style methods accumulate credits in proportion to how much of the graph they
  touch.

The `CreditCheckpoint` list captures the current cut value at specific points in
the solve, paired with the cumulative credits spent so far. This is the data
used to build "performance versus budget" comparisons.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from typing import Dict, Iterator


@dataclass(frozen=True)
class CreditCheckpoint:
    credits_used: int
    cut_value: float
    label: str = ""


@dataclass
class ResourceSnapshot:
    """Aggregate counters for one tracked solver run.

    `evaluation_credits_total` is the headline budget number. The remaining
    fields break that total down into quantum and classical categories and store
    proxy size statistics that help explain where the credits came from.
    """

    evaluation_credits_total: int = 0
    quantum_evaluation_credits: int = 0
    classical_evaluation_credits: int = 0
    quantum_block_updates: int = 0
    quantum_statevector_evaluations: int = 0
    quantum_objective_evaluations: int = 0
    quantum_decode_evaluations: int = 0
    quantum_simulated_amplitudes: int = 0
    quantum_gate_amplitude_ops: int = 0
    quantum_observable_amplitude_ops: int = 0
    quantum_decode_amplitudes_scanned: int = 0
    max_quantum_block_size: int = 0
    classical_exact_block_updates: int = 0
    classical_exact_state_evaluations: int = 0
    classical_exact_tiebreak_state_evaluations: int = 0
    classical_greedy_block_updates: int = 0
    classical_local_search_passes: int = 0
    classical_local_search_node_visits: int = 0
    classical_local_search_edge_touches: int = 0
    classical_local_search_flips: int = 0
    max_classical_block_size: int = 0
    global_exact_state_evaluations: int = 0


class ResourceTracker:
    """Mutable tracker for credits, operation counts, and progress checkpoints."""

    def __init__(self) -> None:
        self.snapshot = ResourceSnapshot()
        self.checkpoints: list[CreditCheckpoint] = []

    def to_dict(self) -> Dict[str, int]:
        return asdict(self.snapshot)

    def record_quantum_credits(self, credits: int = 1) -> None:
        """Add normalized credits to the quantum side of the budget."""
        amount = max(0, int(credits))
        self.snapshot.evaluation_credits_total += amount
        self.snapshot.quantum_evaluation_credits += amount

    def record_classical_credits(self, credits: int) -> None:
        """Add normalized credits to the classical side of the budget."""
        amount = max(0, int(credits))
        self.snapshot.evaluation_credits_total += amount
        self.snapshot.classical_evaluation_credits += amount

    def record_checkpoint(self, cut_value: float, label: str = "") -> None:
        """Store the current cut value against the cumulative credits spent.

        Repeated checkpoints with identical credit usage, value, and label are
        ignored so downstream plots do not fill with duplicate points.
        """
        checkpoint = CreditCheckpoint(
            credits_used=int(self.snapshot.evaluation_credits_total),
            cut_value=float(cut_value),
            label=str(label),
        )
        if self.checkpoints:
            previous = self.checkpoints[-1]
            if (
                previous.credits_used == checkpoint.credits_used
                and abs(previous.cut_value - checkpoint.cut_value) <= 1e-12
                and previous.label == checkpoint.label
            ):
                return
        self.checkpoints.append(checkpoint)

    def record_quantum_block_update(self, block_size: int) -> None:
        self.snapshot.quantum_block_updates += 1
        self.snapshot.max_quantum_block_size = max(self.snapshot.max_quantum_block_size, int(block_size))

    def record_quantum_statevector_evaluation(
        self,
        *,
        num_qubits: int,
        depth: int,
        observable_terms: int,
        evaluation_kind: str = "objective",
        credit_cost: int = 1,
    ) -> None:
        """Charge one quantum evaluation and log rough simulation scale.

        The credit charge is intentionally flat by default. The amplitude-based
        counters provide extra context about the size of that evaluation.
        """
        qubits = max(0, int(num_qubits))
        amplitudes = 1 << qubits
        layers = max(1, int(depth))
        terms = max(1, int(observable_terms))

        self.record_quantum_credits(credit_cost)
        self.snapshot.quantum_statevector_evaluations += 1
        if evaluation_kind == "objective":
            self.snapshot.quantum_objective_evaluations += 1
        elif evaluation_kind == "decode":
            self.snapshot.quantum_decode_evaluations += 1

        self.snapshot.quantum_simulated_amplitudes += amplitudes
        self.snapshot.quantum_gate_amplitude_ops += amplitudes * max(1, qubits * layers)
        self.snapshot.quantum_observable_amplitude_ops += amplitudes * terms

    def record_quantum_decode(self, num_qubits: int, credit_cost: int = 1) -> None:
        """Charge a decode step and count the amplitudes scanned."""
        qubits = max(0, int(num_qubits))
        amplitudes = 1 << qubits
        self.record_quantum_credits(credit_cost)
        self.snapshot.quantum_decode_evaluations += 1
        self.snapshot.quantum_decode_amplitudes_scanned += amplitudes

    def record_classical_exact_block(
        self,
        *,
        block_size: int,
        states: int,
        tiebreak_states: int = 0,
    ) -> None:
        """Charge an exact block solve by the number of states enumerated."""
        self.record_classical_credits(int(states) + int(tiebreak_states))
        self.snapshot.classical_exact_block_updates += 1
        self.snapshot.max_classical_block_size = max(self.snapshot.max_classical_block_size, int(block_size))
        self.snapshot.classical_exact_state_evaluations += max(0, int(states))
        self.snapshot.classical_exact_tiebreak_state_evaluations += max(0, int(tiebreak_states))

    def record_classical_greedy_block(self, block_size: int) -> None:
        self.snapshot.classical_greedy_block_updates += 1
        self.snapshot.max_classical_block_size = max(self.snapshot.max_classical_block_size, int(block_size))

    def record_local_search_pass(
        self,
        *,
        node_visits: int,
        edge_touches: int,
        flips: int,
    ) -> None:
        """Charge a heuristic pass by node visits and log its traversal stats."""
        self.record_classical_credits(node_visits)
        self.snapshot.classical_local_search_passes += 1
        self.snapshot.classical_local_search_node_visits += max(0, int(node_visits))
        self.snapshot.classical_local_search_edge_touches += max(0, int(edge_touches))
        self.snapshot.classical_local_search_flips += max(0, int(flips))

    def record_global_exact(self, states: int) -> None:
        self.record_classical_credits(states)
        self.snapshot.global_exact_state_evaluations += max(0, int(states))


_ACTIVE_TRACKER: ContextVar[ResourceTracker | None] = ContextVar("quahack_active_resource_tracker", default=None)


def get_active_tracker() -> ResourceTracker | None:
    return _ACTIVE_TRACKER.get()


@contextmanager
def tracking_resources(tracker: ResourceTracker | None = None) -> Iterator[ResourceTracker]:
    """Activate a tracker for the current solver call tree.

    The helper functions below are no-ops unless a tracker is active. This lets
    the solver code stay instrumented while benchmark wrappers decide when to
    collect accounting data.
    """
    active = tracker if tracker is not None else ResourceTracker()
    token = _ACTIVE_TRACKER.set(active)
    try:
        yield active
    finally:
        _ACTIVE_TRACKER.reset(token)


def record_quantum_block_update(block_size: int) -> None:
    tracker = get_active_tracker()
    if tracker is not None:
        tracker.record_quantum_block_update(block_size)


def record_quantum_statevector_evaluation(
    *,
    num_qubits: int,
    depth: int,
    observable_terms: int,
    evaluation_kind: str = "objective",
    credit_cost: int = 1,
) -> None:
    tracker = get_active_tracker()
    if tracker is not None:
        tracker.record_quantum_statevector_evaluation(
            num_qubits=num_qubits,
            depth=depth,
            observable_terms=observable_terms,
            evaluation_kind=evaluation_kind,
            credit_cost=credit_cost,
        )


def record_quantum_decode(num_qubits: int, credit_cost: int = 1) -> None:
    tracker = get_active_tracker()
    if tracker is not None:
        tracker.record_quantum_decode(num_qubits, credit_cost=credit_cost)


def record_classical_exact_block(
    *,
    block_size: int,
    states: int,
    tiebreak_states: int = 0,
) -> None:
    tracker = get_active_tracker()
    if tracker is not None:
        tracker.record_classical_exact_block(
            block_size=block_size,
            states=states,
            tiebreak_states=tiebreak_states,
        )


def record_classical_greedy_block(block_size: int) -> None:
    tracker = get_active_tracker()
    if tracker is not None:
        tracker.record_classical_greedy_block(block_size)


def record_local_search_pass(
    *,
    node_visits: int,
    edge_touches: int,
    flips: int,
) -> None:
    tracker = get_active_tracker()
    if tracker is not None:
        tracker.record_local_search_pass(
            node_visits=node_visits,
            edge_touches=edge_touches,
            flips=flips,
        )


def record_global_exact(states: int) -> None:
    tracker = get_active_tracker()
    if tracker is not None:
        tracker.record_global_exact(states)


def record_checkpoint(cut_value: float, label: str = "") -> None:
    tracker = get_active_tracker()
    if tracker is not None:
        tracker.record_checkpoint(cut_value, label=label)
