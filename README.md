# Resource Estimation, Optimization Strategy, and Benchmarking

This document outlines the resource constraints of naïve QAOA, motivates a scalable alternative strategy, and defines a fair benchmarking methodology against classical baselines.

---

# 1. Resource Estimation

## 1.1 Brute-Force QAOA Encoding

For a graph with:

- $N$ nodes  
- $E$ edges  
- QAOA depth $p$  

The conventional one-node-per-qubit encoding requires:

- **Qubits:** $N$
- **Cost layer two-qubit gates:** $O(E)$ per layer
- **Mixer layer single-qubit gates:** $O(N)$ per layer

Total two-qubit gates:

$$
O(pE)
$$

Total single-qubit gates:

$$
O(pN)
$$

Circuit depth depends heavily on graph connectivity and hardware topology.

---

## 1.2 Hardware Constraints (Square-Grid Superconducting Devices)

Modern superconducting devices:

- Have nearest-neighbor connectivity on a square lattice.
- Require SWAP chains to implement non-local interactions.
- Two-qubit gates dominate both:
  - Error accumulation
  - Circuit runtime

If two logical qubits are not adjacent, implementing a two-qubit interaction requires:

- Routing via SWAP gates  
- Each SWAP ≈ 3 CNOT-equivalent gates  
- Routing cost scales with Manhattan distance  

Thus the effective two-qubit gate count becomes:

$$
O(pE \cdot d)
$$

where $d$ is the average routing distance.

For dense graphs:

$$
E = O(N^2)
$$

This makes brute-force QAOA scaling:

$$
O(pN^2)
$$

which quickly becomes infeasible.

---

## 1.3 Why Classical Simulation Fails

Simulating a quantum circuit requires storing:

$$
2^N \text{ complex amplitudes}
$$

Memory requirement:

- $N = 40$ → ~1 TB  
- $N = 50$ → infeasible on standard clusters  

Time complexity also scales exponentially.

Therefore:

- Large circuits are not simulatable classically.
- Large circuits are not executable on current hardware.

---

## 1.4 Practical Two-Qubit Connectivity

On hardware:

- Non-adjacent qubits require SWAP networks.
- Compilation increases circuit depth.
- Two-qubit gate fidelity (~98–99%) compounds multiplicatively.

Realistic constraint:

- ≤ 10–20 qubits
- ≤ 100 two-qubit gates

On current superconducting hardware, practical circuit sizes without heavy error mitigation are typically limited to ~10–20 qubits and on the order of 10² two-qubit gates, as two-qubit gate errors accumulate multiplicatively and dominate fidelity loss.

Thus brute-force QAOA for large graphs is infeasible.

---

# 2. Optimization Strategy

## 2.1 Core Idea: Divide and Conquer

Instead of encoding the entire graph:

1. Partition graph into smaller subgraphs.
2. Solve subgraphs independently.
3. Reconstruct global solution.

This reduces:

- Qubit count per circuit
- Circuit depth
- Two-qubit routing overhead

---

## 2.2 Partition-Based QAOA

We implement:

- Multilevel partitioning
- Region blocks
- Boundary blocks
- Coarse graph construction

Each region is small enough to:

- Fit within hardware constraints
- Limit two-qubit gate count

---

## 2.3 Hybrid Refinement

After solving regions:

- Apply classical refinement (e.g., Burer–Monteiro, greedy)
- Perform block-coordinate descent
- Optionally use warm-start strategies

This creates a **hybrid quantum-classical pipeline**:

1. QAOA-induced deformation (preconditioning)
2. Classical refinement on modified graph
3. Global polish on original graph

---

## 2.4 Motivation

Why not brute-force QAOA?

- Hardware-limited
- SWAP overhead destroys depth advantage
- Noise accumulates

Why not purely classical?

- QAOA-induced deformation may reshape optimization landscape
- Hybrid methods may escape classical local minima

---

## 2.5 Limitations

- Partition quality strongly affects performance.
- Boundary size can dominate runtime.
- Improvements may be modest if classical baseline is already strong.
- No formal guarantee of quantum advantage.

---

# 3. Benchmarking Strategy

## 3.1 Classical Baseline

We use strong classical references:

- Burer–Monteiro (BM) seed sweep
- BM + greedy polish
- Optionally: NetworkX `one_exchange()` heuristic

The baseline must be competitive.

---

## 3.2 Fair Comparison Criteria

To benchmark fairly:

- Evaluate all assignments on the **original graph**
- Report:
  - Final cut value
  - Runtime
  - Resource accounting (if applicable)
- Use identical stopping criteria where possible

Avoid:

- Comparing weak classical heuristics
- Using inconsistent seeds
- Allowing silent post-processing improvements

---

## 3.3 Runtime vs Accuracy Trade-offs

Classical BM:

- Fast
- Highly parallelizable
- Strong local search performance

Partitioned QAOA:

- Lower qubit requirement
- Higher overhead per region
- Sensitive to hyperparameters

---

## 3.4 Bottlenecks

Major bottlenecks:

- Boundary block optimization
- Large region blocks
- Two-qubit gate depth in cost layer
- Classical refinement convergence

---

## 3.5 Directions for Improvement

- Better partitioning heuristics
- Adaptive region sizing
- Hardware-aware layout optimization
- Error mitigation strategies
- Parameter transfer learning between subgraphs

---

# Conclusion

Brute-force QAOA is infeasible for large graphs due to:

- Qubit count
- Two-qubit routing overhead
- Noise accumulation
- Classical simulation limits

Partition-based hybrid QAOA provides a scalable alternative:

- Reduces quantum resource requirements
- Enables hardware-compatible subproblems
- Integrates strong classical refinement

Benchmarking must remain rigorous and transparent to meaningfully assess improvement over classical baselines.