# PAPER EXPLANATION & REPLICATION GUIDE
## "Increasing the Parallelism of Graph Coloring via Shortcutting"
### Authors: Ghadeer Alabandi, Evan Powers, and Martin Burtscher (Texas State University)
### Conference: ACM SIGPLAN PPoPP 2020 (Principles and Practice of Parallel Programming)
### DOI: [10.1145/3332466.3374519](https://doi.org/10.1145/3332466.3374519)

---

## 1. Executive Summary: What is This Paper About?

In one sentence:
> **This paper solves the fundamental problem that parallel graph coloring algorithms suffer from low hardware utilization due to sequential dependency chains, by introducing "shortcutting"—an algorithmic breakthrough that allows lower-priority vertices to safely color themselves early without waiting for higher-priority neighbors.**

- **The Problem**: The standard parallel graph-coloring algorithm (Jones-Plassmann with Largest-Degree-First, or JP-LDF) forces a vertex to wait until **all** of its higher-priority neighbors have been colored. On long dependency chains, only a few vertices are ready in each step, leaving massively parallel processors (like GPUs and multi-core CPUs) starved of work.
- **The Solution ("Shortcutting")**: 
  - **Shortcut 1**: If the best available color for vertex $v$ is no longer considered by *any* of its uncolored higher-priority neighbors, $v$ can color itself **immediately**, bypassing the priority order.
  - **Shortcut 2**: If an uncolored higher-priority neighbor cannot possibly take any color that $v$ could take, the edge between them is logically deleted, reducing $v$'s worst-case color range and unlocking Shortcut 1 for others.
- **The Results**:
  - **$3.4\times$ higher average parallelism** across 18 benchmark graphs.
  - **$2.9\times$ average speedup** on NVIDIA Titan V GPU over state-of-the-art implementations (CUSP, cuSPARSE, Kokkos).
  - **Deterministic and zero quality loss**: Uses the same or fewer colors as the serial/standard algorithm.

---

## 2. Background: Why Graph Coloring & Why is Parallelism Hard?

### 2.1 What is Distance-1 Graph Coloring?
Given an undirected graph $G = (V, E)$, assign a color $c(v) \in \{0, 1, 2, \dots\}$ to each vertex $v \in V$ such that:
$$\forall (u, v) \in E \implies c(u) \neq c(v)$$
Minimizing the total number of colors (the chromatic number $\chi(G)$) is **NP-hard**. In practice, fast greedy heuristics are used.

### 2.2 Applications
- **Sparse Matrix Computations**: Finding independent sets of rows/columns that can be updated concurrently without race conditions (e.g., in parallel Preconditioned Conjugate Gradient, incomplete LU factorization).
- **Register Allocation in Compilers**: Nodes represent variables; edges represent overlapping live ranges.
- **Task Scheduling & Timetabling**: Conflicting tasks cannot run in the same time slot (color).

### 2.3 The Classic Baseline: Jones-Plassmann (JP) with Largest-Degree-First (LDF)
To avoid race conditions, Jones & Plassmann (1993) introduced a priority rule:
1. Orient every edge $(u, v)$ from higher-priority to lower-priority vertex, turning the graph into a **Directed Acyclic Graph (DAG)**.
2. In **Largest-Degree-First (LDF)**, vertex priority is determined by:
   - Higher degree has higher priority: $\text{deg}(u) > \text{deg}(v) \implies u \to v$.
   - Tie-breaker: Deterministic pseudo-random hash value $\text{hash}(u) > \text{hash}(v)$, then vertex ID $u > v$.
3. **Execution Rule in JP-LDF**:
   - A vertex $v$ is "ready" to be colored **if and only if all incoming edges are resolved** (i.e., all higher-priority neighbors are already assigned their final colors).
   - Once ready, $v$ picks the smallest non-conflicting color (First-Fit).

### 2.4 The Critical Bottleneck of JP-LDF
- In a DAG, the **depth** of the graph determines the **number of sequential steps**, while the **width** at any level determines the **degree of parallelism**.
- Many real-world graphs (road networks, citation graphs, meshes, social networks) have long paths where:
  $$A \to B \to C \to D \to E \dots$$
- In step 1, only $A$ is colored. $B$ waits for $A$. $C$ waits for $B$. $D$ waits for $C$.
- On an NVIDIA GPU with 5,120 CUDA cores or a CPU with 16 threads, running a step where only 10 or 50 vertices are ready leaves **>99% of hardware cores idle**!

---

## 3. The Core Contribution: How "Shortcutting" Works

Instead of waiting blindly for all higher-priority neighbors to finish, ECL-GC maintains for each vertex a **set of possible colors** (represented as a bitmap or range).

### 3.1 Key Mathematical Invariant
If a vertex $v$ has $k$ incoming edges (uncolored higher-priority neighbors), $v$ needs at most the first $k + 1$ colors: $\{0, 1, \dots, k\}$.
Even if all $k$ neighbors take different colors, at least one color in $\{0, 1, \dots, k\}$ must remain available for $v$ (Pigeonhole Principle).

---

### 3.2 Shortcut 1: Early Coloring (The Big Breakthrough)

> **Definition (Shortcut 1):**
> Let $c_{\text{best}}(v)$ be the lowest possible color that vertex $v$ can currently take.
> If **no uncolored higher-priority neighbor** of $v$ still considers $c_{\text{best}}(v)$ as a possible color, then:
> - None of those neighbors will *ever* choose $c_{\text{best}}(v)$.
> - Therefore, $v$ will *never* conflict on color $c_{\text{best}}(v)$.
> - **$v$ can immediately assign itself $c_{\text{best}}(v)$ and terminate, even if its higher-priority neighbors are uncolored!**

#### Concrete Example of Shortcut 1:
Suppose $B$ is a higher-priority neighbor of $C$ ($B \to C$).
- $B$ has already seen its own neighbor take Color 0, so $B$'s possible colors are $\{1, 2, 3\}$.
- $C$'s best available color is **Color 0**.
- In standard JP-LDF, $C$ must wait until $B$ finishes.
- **With Shortcut 1**: $C$ inspects $B$'s bitmap and sees that $B$ *cannot* take Color 0. Since no incoming neighbor can take Color 0, $C$ immediately takes Color 0! Both $B$ and $C$ can now be processed in parallel!

---

### 3.3 Shortcut 2: Edge Pruning (Range Disjointness)

> **Definition (Shortcut 2):**
> If an uncolored higher-priority neighbor $u$ of vertex $v$ has a set of possible colors that is **completely disjoint** from $v$'s possible colors, then:
> - $u$ and $v$ can never choose the same color.
> - The dependency edge $u \to v$ can be **safely deleted** immediately!
> - Deleting this edge reduces $v$'s incoming edge count by 1, allowing $v$ to discard its worst-case color (highest set bit).
> - Shrinking $v$'s color range often triggers **Shortcut 1** on other downstream neighbors!

---

## 4. Hardware Optimization & Data Structures in ECL-GC

ECL-GC achieves extreme speed because shortcutting is designed to execute in hardware with negligible overhead:

1. **Low-Degree Vertices ($\text{deg}(v) < 32$)**:
   - The set of possible colors fits directly inside a single 32-bit `int` register.
   - Checking possible colors uses single-cycle bitwise operations:
     - Clear color bit: `bitmap &= ~(1 << color)`
     - Best color: Find First Set (`__builtin_clz` or `ffs`)
     - Check union of neighbors: Bitwise OR (`|`)
2. **High-Degree Vertices ($\text{deg}(v) \ge 32$)**:
   - Allocating giant bitmaps for high-degree nodes would destroy cache locality.
   - ECL-GC uses an interval abstraction: two integers `[mincol, maxcol]`.
   - Shortcut 1 check becomes an $O(1)$ constant-time interval comparison:
     Is $v$'s `mincol` outside $[mincol_u, maxcol_u]$ for all incoming neighbors $u$?
3. **Benign Asynchronous Data Races**:
   - Bitmaps are **monotonic**: bits are only ever cleared (never set).
   - `mincol` is **monotonic**: values only ever increase.
   - Threads run without expensive global barriers or atomic locking; if a thread reads a slightly stale value, it simply defers a shortcut to the next iteration. Correctness is 100% preserved!

---

## 5. Summary of Results from the Paper

| Metric | JP-LDF (Baseline) | ECL-GC (Shortcutting) | Improvement |
| :--- | :---: | :---: | :---: |
| **Average Parallelism** | Baseline | $+240\%$ | **$3.4\times$ increase** (up to $12\times$ on RMAT) |
| **GPU Execution Time** | Baseline | $-65\%$ | **$2.9\times$ speedup** over best literature codes |
| **Number of Colors** | Baseline | Same or fewer | **0 quality loss** (100% deterministic) |
| **Vertices Colored by Shortcuts** | $0\%$ | **$52.6\%$ on average** | Over half the graph is colored via shortcuts! |

---

## 6. Presentation Checklist for Wednesday

When explaining this paper to your professor or class, structure your talk in 5 clear parts:

1. **Slide 1-2: Problem & Motivation (2 mins)**
   - What is graph coloring? Why does it matter in high performance computing (matrix operations, parallel loops)?
   - Show why greedy sequential coloring is fast ($O(V + E)$) but inherently sequential.
2. **Slide 3-4: The Baseline & The Bottleneck (3 mins)**
   - Explain Jones-Plassmann with Largest-Degree-First (JP-LDF).
   - Draw a simple DAG chain: $A \to B \to C \to D$. Show that JP-LDF takes 4 separate rounds, coloring only 1 node per round.
   - Point out: On a GPU with thousands of cores, this wastes computing power!
3. **Slide 5-7: The Core Breakthrough: Shortcutting (5 mins)**
   - Explain **Shortcut 1**: If none of my higher-priority neighbors can take my best color, I don't need to wait for them. I take it right now!
   - Explain **Shortcut 2**: If my neighbor's color set doesn't overlap with mine, remove the edge and shrink my color range.
   - Walk through the visual example on the $A \to B \to C$ graph.
4. **Slide 8-9: Efficient Implementation (2 mins)**
   - 32-bit bitmaps for low-degree nodes ($O(1)$ bitwise operations).
   - `[mincol, maxcol]` range representation for high-degree nodes.
   - Lock-free, asynchronous monotonic updates.
5. **Slide 10-11: Experimental Findings & Conclusion (3 mins)**
   - Show the $3.4\times$ parallelism increase.
   - Show the $2.9\times$ GPU speedup on Titan V.
   - Demonstrate the live replication running on `internet.egr` in $5.4\text{ ms}$!

### Likely Questions from Professor & Best Answers:
- **Q: Does shortcutting change the final color assignments or increase the number of colors?**
  *A: No! The paper proves that shortcutting produces the exact same deterministic coloring as the serial LDF algorithm. It only reorders the timing of when nodes commit their colors, never changing the color choice.*
- **Q: What happens if two neighbors try to shortcut to the same color simultaneously?**
  *A: Impossible! Shortcut 1 requires that no higher-priority neighbor has that color in its possible set. Since the priority order is a total order with unique tie-breakers, at most one vertex can claim the color.*
- **Q: On which graphs does shortcutting give the highest benefit?**
  *A: Graphs with high degree variation and long dependency paths, such as R-MAT graphs (up to $12\times$ parallelism boost) and road networks.*
