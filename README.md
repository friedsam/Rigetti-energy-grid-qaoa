# Resource Estimation, Optimization Strategy, and Benchmarking

This document outlines the resource constraints of naïve QAOA, motivates a scalable alternative strategy, and defines a fair benchmarking methodology against classical baselines.

---

# 1. Resource Estimation

## 1.1 Brute-Force QAOA Encoding

For a graph with:

- $N$ nodes  
- $E$ edges  
- QAOA depth - $p$ 

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

Simulating a general quantum circuit requires storing:

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

Simulating a noisy quantum circuit is possible in polynomial time, but without extra structure incurs a quadratic time cost penalty. This limits the usefulness of NISQ computing to problems, like optimization, where the problem size is large enough and the problem is important enough that a polynomial speedup matters.

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
2. Calculate a heuristic for each region.
3. Solve subgraphs independently with the method most suited.
4. Reconstruct global solution.

This reduces:

- Qubit count per circuit
- Circuit depth
- Two-qubit routing overhead

---

## 2.2 Partitioning

We implement a MITES based method for determining regions:
1. Seed regions from weighted modularlity communities.
2. Split oversized regions with Kernighan-Lin bisection.
3. Merge small regions with their neighbors.
---

## 2.3 Heuristic Solver Choice

We compute a heurisitic score based on the presence of cycles and the density of the region.

We then cluster these scores with node count to pick the best solver for a given region.

The heuristic is motivated by quantum methods having a advantage in complicated optimization landscapes and preconditioning being more efficent at higher node counts.

---

## 2.4 Motivation

Why not brute-force QAOA?

- Hardware-limited
- SWAP overhead destroys depth advantage
- Noise accumulates

Why not purely classical?

- QAOA-induced deformation may reshape optimization landscape
- Hybrid methods may escape classical local minima

The best results can be found from choosing the optimal method as often as possible

---

## 2.5 Limitations

- Light cones were too computationally expensive, we only had enough time to run standard qaoa on actual hardware.
- Boundary size can dominate runtime.
- Improvements may be modest if classical baseline is already strong.
- No formal guarantee of quantum advantage.

---

# 3. Benchmarking Strategy

## 3.1 Classical Baseline

We use strong classical references:

- Burer–Monteiro (BM) seed sweep
- BM + greedy polish

The baseline must be competitive.

---

## 3.2 Fair Comparison Criteria

We use a credit based system that attempts to account for the true work budget of a algorithm.

QAOA consistently outperforms classical methods.

More details can be found within the Modular QAOA folder in the benchmarking file.

Only simulator runs were benchmarked fully.


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
## 4 Hardware Execution

Standard QAOA was run on Ankaa-3 hardware for small sub-regions resulting in similar results to the simulator.

Preconditioning was attempted on hardware but, we made miscalculations with the expected runtime causing it to fail.

The graph with a significant portion run through QAOA on hardware achieved 99.6% of the optimal value(7071.1).

Overall, these results are highly encouraging and indicate potential for this idea to be further extended in the future with more variants of QAOA.

The overall results are available in the results folder.

## 5 Notes on the Repository

Notebooks containing simulator based QAOA and partitioning are available in notebooks. The notebooks used on the real hardware are in the folder indicating real hardware. The error mitigation part of the repository wasn't used as none of the error was strong enough to warrant the extra computational cost, but we thought it was really cool. It is based on Q-Ctrl's Fire Opal. Currently, the pulse level is not functional but the rest can be wired into the modular QAOA solver. It implements Zero Noise Estimation and a host of compilation techniques to mitigate and supress error. It was built to take in the calibration data from Ankaa-3. It is probably somewhat broken now due to merge conflicts.
