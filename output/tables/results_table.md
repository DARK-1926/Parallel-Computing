# PPoPP 2020 Shortcutting Replication - Results Table

### 1. Paper Table 4 Replication (Parallelism & Dependency Steps)

| Metric | Paper Published Result | Our Device Replicated Result | Agreement |
| :--- | :---: | :---: | :---: |
| **Graph Name** | `internet` | `internet.egr` | 100% Identical |
| **Vertices** | 124,651 | 124,651 | Exact Match |
| **Edges** | 387,240 | 387,240 | Exact Match |
| **Steps w/o Shortcuts** | 27 | 28 | ~96.4% Match |
| **Steps with Shortcuts** | 13 | 13 | **100% Exact Match** |
| **Avg Parallelism (w/o)** | 4,616.7 nodes/step | 4,451.8 nodes/step | ~96.4% Match |
| **Avg Parallelism (with)** | 9,588.5 nodes/step | 9,588.5 nodes/step | **100% Exact Match** |
| **Parallelism Boost** | **2.08x** | **2.15x** | **Replicated (~103%)** |
| **Colors Used** | 8 | 8 | **Exact Match (No Quality Loss)** |

### 2. Device Thread Scaling (AMD Ryzen 7 7735HS)

| Threads | Runtime (ms) | Speedup | Throughput (Medges/s) | Throughput (Mnodes/s) | Verified |
| :---: | :---: | :---: | :---: | :---: | :---: |
|  1 |   13.26 ms |  1.00x |  29.20 Medges/s |   9.40 Mnodes/s | PASSED |
|  2 |    7.80 ms |  1.70x |  49.65 Medges/s |  15.98 Mnodes/s | PASSED |
|  4 |    6.04 ms |  2.20x |  64.10 Medges/s |  20.63 Mnodes/s | PASSED |
|  8 |    7.03 ms |  1.89x |  55.10 Medges/s |  17.74 Mnodes/s | PASSED |
| 16 |    8.08 ms |  1.64x |  47.92 Medges/s |  15.43 Mnodes/s | PASSED |
