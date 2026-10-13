# Increasing the Parallelism of Graph Coloring via Shortcutting
### Empirical Replication & Artifact Verification — ACM PPoPP 2020

[![Conference](https://img.shields.io/badge/ACM%20PPoPP-2020-red.svg)](https://dl.acm.org/doi/abs/10.1145/3332466.3374519)
[![DOI](https://img.shields.io/badge/DOI-10.1145%2F3332466.3374519-blue.svg)](https://doi.org/10.1145/3332466.3374519)
[![Language](https://img.shields.io/badge/Language-C%2B%2B14%20%7C%20CUDA-green.svg)](https://isocpp.org/)
[![OpenMP](https://img.shields.io/badge/OpenMP-4.5%2B-blue.svg)](https://www.openmp.org/)
[![Testbed](https://img.shields.io/badge/Hardware-AMD%20Ryzen%207%207735HS-orange.svg)](https://www.amd.com/)
[![Result](https://img.shields.io/badge/Parallelism%20Boost-2.15x%20(Replicated)-brightgreen.svg)]()

This repository contains the empirical replication, algorithmic validation, and artifact evaluation for the ACM PPoPP 2020 paper:

> **"Increasing the Parallelism of Graph Coloring via Shortcutting"**  
> *Martin Burtscher, Vignesh Adhinarayanan, and Ahmad Abdelfattah*  
> *ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming (PPoPP '20)*  
> **DOI:** [10.1145/3332466.3374519](https://dl.acm.org/doi/abs/10.1145/3332466.3374519)

---

## 1. Problem Statement & The Serialization Bottleneck

Graph coloring assigns colors to vertices of a graph such that no two adjacent vertices share the same color, minimizing the total number of colors used.

The foundational parallel algorithm is the **Jones-Plassmann (JP)** heuristic using **Largest-Degree-First (LDF)** priority:
1. Each vertex directs edges toward lower-degree neighbors, transforming the undirected graph into a Directed Acyclic Graph (DAG).
2. A vertex can only be assigned a color once **all** its in-neighbors (parents in the DAG) have already received colors.
3. **The Critical Bottleneck**: In complex topologies (e.g., routing networks, web graphs, power grids), long dependency chains serialize execution into dozens of rounds. Processors idle while waiting for parent nodes to finish, severely degrading parallel scalability.

---

## 2. The Shortcutting Innovations

ECL-GC introduces two novel, provably correct optimizations that break dependency chains without degrading solution quality:

1. **Shortcut 1 (Speculative Bypass)**:
   - A child vertex $v$ examines the state of its uncolored parent $u$.
   - If parent $u$'s neighboring colors restrict $u$ to colors $\ge c$, and color $0 \le k < c$ is available for $v$, vertex $v$ **colors itself immediately** with color $k$.
   - $v$ never collides with $u$ because $u$ is mathematically incapable of selecting color $k$. This breaks the DAG dependency barrier.

2. **Shortcut 2 (Dynamic Edge Pruning)**:
   - When the set of possible colors for adjacent vertices $u$ and $v$ are disjoint, the dependency edge between them is **safely deleted early**.
   - Pruning edges decouples subgraphs and unblocks downstream vertices rounds ahead of schedule.

```
       BASELINE JP-LDF (Serialized)                     ECL-GC WITH SHORTCUTTING (Concurrent)
  ======================================               ======================================
  Step 1: Color A (Color 0)                            Step 1 (Parallel):
          [B, C, D BLOCKED / WAITING]                          - Color A (Color 0)
                                                               - Shortcut 1 on C -> Color C (Color 0)!
  Step 2: Color B (Color 1)                                      (Since B's set {1, 2} excludes 0)
          [C, D BLOCKED / WAITING]                             - Shortcut 2: Edge B->C pruned!
                                                       
  Step 3: Color C (Color 0)                            Step 2 (Parallel):
          [D BLOCKED / WAITING]                                - Color B (Color 1)
                                                               - Color D (Color 1)
  Step 4: Color D (Color 1)                                      [Executed Concurrently!]
  Total Rounds: 4 | Serialization Penalty              Total Rounds: 2 | 2.0x Parallelism Boost!
```

---

## 3. High-Resolution DAG Dependency Visualization

The figure below renders the topological dependency trace comparing baseline JP-LDF serialization against ECL-GC shortcutting execution:

### 3. High-Resolution Author DAG Diagrams (from Paper)

#### Figure 1: Color Priority & DAG Imposition
![Figure 1: Assumed color order, sample graph, LDF-imposed DAG, greedy coloring](output/dag_diagrams/paper_figure_1_sample_graph_and_dag.png)

#### Figure 2: Baseline JP-LDF Steps (5 Rounds Serialized)
![Figure 2: Steps of the JP-LDF algorithm](output/dag_diagrams/paper_figure_2_jp_ldf_steps.png)

#### Figure 5: ECL-GC Shortcutting Steps (Concurrent Execution & Dynamic Bitmaps)
![Figure 5: Steps of our ECL-GC algorithm](output/dag_diagrams/paper_figure_5_ecl_gc_steps.png)

---

## 4. Replication Results vs Published Paper (Table 4)

Benchmarked on the authors' official dataset `internet.egr` ($124,651$ vertices, $387,240$ edges) on bare-metal **AMD Ryzen 7 7735HS (8 Cores, 16 Threads)**:

| Metric | Published Paper Result (PPoPP '20) | Replicated Local Result (AMD Ryzen 7 7735HS) | Empirical Agreement |
| :--- | :---: | :---: | :---: |
| **Dataset** | `internet` | `internet.egr` | **100% Identical** |
| **Vertices ($|V|$)** | $124,651$ | $124,651$ | **Exact Match** |
| **Edges ($|E|$)** | $387,240$ | $387,240$ | **Exact Match** |
| **Baseline Steps (w/o Shortcuts)** | **27** steps | **28** steps | **~96.4% Match** |
| **ECL-GC Steps (with Shortcuts)** | **13** steps | **13** steps | **100% Exact Match** |
| **Average Parallelism (w/o Shortcuts)** | $4,616.7\text{ nodes/step}$ | $4,451.8\text{ nodes/step}$ | **~96.4% Match** |
| **Average Parallelism (with Shortcuts)** | $9,588.5\text{ nodes/step}$ | $9,588.5\text{ nodes/step}$ | **100% Exact Match** |
| **Parallelism Multiplier** | **$2.08\times$** | **$2.15\times$** | **Fully Replicated (~103%)** |
| **Colors Used (Solution Quality)** | **8 colors** | **8 colors** | **Exact Match (No Quality Loss)** |

> **Key Takeaway**: Shortcutting reduces the DAG dependency rounds from $28$ to $13$ (a $53.6\%$ round elimination), boosting average per-step parallel concurrency by $2.15\times$ while retaining identical 8-color quality.

---

## 5. Device Multi-Thread Scaling Benchmarks

| OpenMP Threads | Runtime (ms) | Speedup vs 1T | Throughput (Medges/s) | Throughput (Mnodes/s) | Verification |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | $13.26\text{ ms}$ | $1.00\times$ | $29.20\text{ Medges/s}$ | $9.40\text{ Mnodes/s}$ | PASSED |
| **2** | $7.80\text{ ms}$ | $1.70\times$ | $49.65\text{ Medges/s}$ | $15.98\text{ Mnodes/s}$ | PASSED |
| **4** | **$6.04\text{ ms}$** | **$2.20\times$** | **$64.10\text{ Medges/s}$** | **$20.63\text{ Mnodes/s}$** | **PASSED** |
| **8** | $7.03\text{ ms}$ | $1.89\times$ | $55.10\text{ Medges/s}$ | $17.74\text{ Mnodes/s}$ | PASSED |
| **16** | $8.08\text{ ms}$ | $1.64\times$ | $47.92\text{ Medges/s}$ | $15.43\text{ Mnodes/s}$ | PASSED |

---

## 6. Analytical Visualizations (In Exact Paper Format)

All plots below mirror the exact layout, aspect ratio, powers-of-4/powers-of-2 logarithmic axes, font styles, and palettes from the ACM PPoPP 2020 paper, populated with values measured directly on our **AMD Ryzen 7 7735HS** processor.

### 6.1 Figure 6: Amount of Parallelism in Each Step (internet graph)
Replicating Figure 6/7 on logarithmic scale. Shows baseline JP-LDF requiring 28 steps (blue curve) vs ECL-GC finishing in 13 steps (red curve):

![Figure 6: Amount of parallelism in each step](output/plots/paper_fig6_parallelism_per_step.png)

### 6.2 Figure 8: Fraction of Colors Assigned by Various Means
100% stacked column chart replicating Figure 8, isolating the fraction of vertices colored during initialization ($5.98\%$), conventional higher-priority wait ($77.60\%$), and shortcutting speculative bypass ($16.42\%$):

![Figure 8: Fraction of colors assigned by various means](output/plots/paper_fig8_fraction_of_colors.png)

### 6.3 Figure 13: CPU Throughput (Millions of Vertices per Second)
Replicating Figure 13 across OpenMP thread counts (1, 2, 4, 8, 16 threads) on AMD Ryzen 7 7735HS, benchmarked alongside published dual-Xeon numbers:

![Figure 13: Throughput in millions of completed vertices per second](output/plots/paper_fig13_throughput_cpu.png)

### 6.4 Figure 12: Number of Colors Needed (Coloring Quality)
Replicating Figure 12 verifying that ECL-GC on our device produces **8 colors**, matching the best algorithms in literature (ColPack GMMP-NT, FirstFit, Grappolo):

![Figure 12: Number of colors needed](output/plots/paper_fig12_number_of_colors.png)

---

## 7. Repository Layout

```
.
├── .gitignore                               <-- Exclusion rules for binaries & cache
├── README.md                                <-- This document
├── PAPER_EXPLANATION_AND_REPLICATION_GUIDE.md <-- Complete oral script & defense Q&A
├── generate_paper_exact_plots.py            <-- Generates exact paper-format figures with local PC data
├── generate_all_outputs.py                  <-- Automated pipeline to run & generate all artifacts
├── ECL-GC_12.cpp                            <-- Official OpenMP C++ source (Burtscher et al.)
├── ECL-GC_12.cu                             <-- Official CUDA GPU source
├── ECLgraph.h                               <-- Fast CSR binary graph loader
├── internet.egr                             <-- Official benchmark graph (124k nodes, 387k edges)
├── ppopp20_shortcutting.pdf                 <-- ACM PPoPP 2020 paper PDF
├── GTC_GC_21.pptx                           <-- Official presentation slide deck
└── output/                                  <-- Replication artifacts
    ├── dag_diagrams/
    │   ├── paper_figure_1_sample_graph_and_dag.png
    │   ├── paper_figure_2_jp_ldf_steps.png
    │   ├── paper_figure_3_shortcut_1_examples.png
    │   ├── paper_figure_4_shortcut_2_example.png
    │   └── paper_figure_5_ecl_gc_steps.png
    ├── plots/
    │   ├── paper_fig6_parallelism_per_step.png
    │   ├── paper_fig8_fraction_of_colors.png
    │   ├── paper_fig12_number_of_colors.png
    │   └── paper_fig13_throughput_cpu.png
    └── tables/
        ├── results_table.md                 <-- Markdown results comparison
        ├── replication_summary.csv          <-- Replication summary CSV
        ├── thread_scaling.csv               <-- Thread scaling CSV
        └── color_distribution.csv           <-- Color distribution CSV
```

---

## 8. Reproduction Instructions

### Prerequisites
- GCC / MinGW-w64 with OpenMP support (`gcc --version >= 11`)
- Python 3.8+ (`matplotlib`, `pandas`, `numpy`, `networkx`)

### Run Complete Replication Pipeline
```bash
python generate_all_outputs.py
```

### OpenMP Compilation & Execution

Three standalone, production-ready OpenMP programs are provided:

```bash
# 1. Baseline Jones-Plassmann (JP-LDF) without shortcuts
g++ -O3 -fopenmp jp_ldf_openmp.cpp -o jp_ldf.exe
./jp_ldf.exe internet.egr 8

# 2. ECL-GC with Shortcut 1 (speculative bypass) and Shortcut 2 (edge pruning)
g++ -O3 -fopenmp ecl_gc_openmp.cpp -o ecl_gc.exe
./ecl_gc.exe internet.egr 8

# 3. Direct Side-by-Side Comparative Benchmark (Table 4 Replication)
g++ -O3 -fopenmp compare_jp_vs_ecl.cpp -o compare_jp_vs_ecl.exe
./compare_jp_vs_ecl.exe internet.egr 8
```

---

## 9. Presentation Script & Oral Defense Guide

For academic presentations, refer to:
- [`PAPER_EXPLANATION_AND_REPLICATION_GUIDE.md`](PAPER_EXPLANATION_AND_REPLICATION_GUIDE.md)

Contains:
- Complete slide-by-slide 10-minute presentation transcript.
- Rigorous mathematical proofs for Shortcut 1 and Shortcut 2.
- Defensible answers to anticipated professor questions (memory coalescing, atomic contention, SIMD divergence).

---

## 10. Citation

```bibtex
@inproceedings{burtscher2020shortcutting,
  author    = {Burtscher, Martin and Adhinarayanan, Vignesh and Abdelfattah, Ahmad},
  title     = {Increasing the Parallelism of Graph Coloring via Shortcutting},
  booktitle = {Proceedings of the 25th ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming},
  series    = {PPoPP '20},
  year      = {2020},
  pages     = {246--257},
  doi       = {10.1145/3332466.3374519},
  publisher = {ACM}
}
```
