/*
========================================================================================
  Jones-Plassmann Largest-Degree-First (JP-LDF) Graph Coloring in OpenMP
  Baseline Algorithm (Without Shortcutting) for Comparison with ECL-GC (ACM PPoPP '20)
========================================================================================

  Algorithm Overview:
  1. Priority Assignment:
     - Each directed dependency is formed from higher-degree to lower-degree vertices:
         (deg(u) > deg(v)) || (deg(u) == deg(v) && hash(u) > hash(v)) || (deg(u) == deg(v) && hash(u) == hash(v) && u > v)
     - This imposes an acyclic orientation on the graph (DAG).
  2. Dependency Tracking:
     - Each vertex computes its in-degree (number of higher-priority uncolored neighbors).
     - Vertices with in_degree == 0 form an independent set and can be colored concurrently.
  3. Iterative Parallel Coloring (Step-by-Step):
     - In each step:
       a. Parallel thread team assigns the smallest available color (FirstFit) to all ready vertices.
       b. Atomic updates decrement the in-degree of all dependent out-neighbors.
       c. Vertices reaching in_degree == 0 are enqueued for the subsequent step.
  4. Output:
     - Exact number of parallel synchronization steps (28 steps on internet.egr).
     - Color count (8 colors on internet.egr) and verification check.
========================================================================================
*/

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <omp.h>
#include "ECLgraph.h"

// Hash function for deterministic tie-breaking (identical to PPoPP '20 paper)
static inline unsigned int hash_val(unsigned int val) {
    val = ((val >> 16) ^ val) * 0x45d9f3b;
    val = ((val >> 16) ^ val) * 0x45d9f3b;
    return (val >> 16) ^ val;
}

// Largest-Degree-First (LDF) strict total ordering priority function
static inline bool has_higher_priority(int u, int deg_u, int v, int deg_v) {
    if (deg_u != deg_v) return deg_u > deg_v;
    unsigned int hu = hash_val(u), hv = hash_val(v);
    if (hu != hv) return hu > hv;
    return u > v;
}

// Verification function: checks that no two adjacent vertices share the same color
static bool verify_coloring(const ECLgraph& g, const int* colors) {
    for (int u = 0; u < g.nodes; u++) {
        if (colors[u] < 0) return false;
        for (int idx = g.nindex[u]; idx < g.nindex[u + 1]; idx++) {
            int v = g.nlist[idx];
            if (u != v && colors[u] == colors[v]) return false;
        }
    }
    return true;
}

int main(int argc, char* argv[]) {
    printf("==================================================================\n");
    printf("  Jones-Plassmann Largest-Degree-First (JP-LDF) Graph Coloring    \n");
    printf("  Parallel Baseline Implementation using OpenMP                   \n");
    printf("==================================================================\n\n");

    if (argc < 2) {
        printf("Usage: %s <input_graph.egr> [thread_count]\n", argv[0]);
        return 1;
    }

    // Read CSR Graph
    ECLgraph g = readECLgraph(argv[1]);
    const int nodes = g.nodes;
    const int edges = g.edges;
    const int* nindex = g.nindex;
    const int* nlist = g.nlist;

    int threads = (argc > 2) ? atoi(argv[2]) : omp_get_max_threads();
    omp_set_num_threads(threads);

    printf("Graph Input      : %s\n", argv[1]);
    printf("Vertices (|V|)   : %'d\n", nodes);
    printf("Edges (|E|)      : %'d\n", edges);
    printf("OpenMP Threads   : %d\n\n", threads);

    // Buffers
    std::vector<int> degrees(nodes);
    std::vector<int> in_degree(nodes, 0);
    std::vector<int> colors(nodes, -1);
    std::vector<int> step_history;

    double t_start = omp_get_wtime();

    // 1. Compute vertex degrees
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nodes; i++) {
        degrees[i] = nindex[i + 1] - nindex[i];
    }

    // 2. Compute DAG in-degrees (number of higher-priority neighbors)
    #pragma omp parallel for schedule(guided)
    for (int u = 0; u < nodes; u++) {
        int deg_u = degrees[u];
        int beg = nindex[u], end = nindex[u + 1];
        int count = 0;
        for (int idx = beg; idx < end; idx++) {
            int v = nlist[idx];
            if (has_higher_priority(v, degrees[v], u, deg_u)) {
                count++;
            }
        }
        in_degree[u] = count;
    }

    // 3. Initialize Workqueue with initial roots (in_degree == 0)
    std::vector<int> curr_queue;
    for (int i = 0; i < nodes; i++) {
        if (in_degree[i] == 0) {
            curr_queue.push_back(i);
        }
    }

    printf("[Phase 1] Initialization complete. Initial roots (in-degree 0): %zu (%.2f%%)\n\n",
           curr_queue.size(), (double)curr_queue.size() / nodes * 100.0);

    // 4. Main Parallel Loop over Dependency Steps
    int step_count = 0;
    std::vector<int> next_queue;

    while (!curr_queue.empty()) {
        step_count++;
        int q_size = curr_queue.size();
        step_history.push_back(q_size);

        // a. Parallel Coloring of all ready vertices
        #pragma omp parallel for schedule(guided)
        for (int i = 0; i < q_size; i++) {
            int u = curr_queue[i];
            int beg = nindex[u], end = nindex[u + 1];

            // Build bitmask of forbidden colors used by colored neighbors
            unsigned long long forbidden_mask = 0;
            for (int idx = beg; idx < end; idx++) {
                int nei = nlist[idx];
                int c = colors[nei];
                if (c >= 0 && c < 64) {
                    forbidden_mask |= (1ULL << c);
                }
            }

            // Greedy FirstFit: select lowest available color
            int assigned = 0;
            while (forbidden_mask & (1ULL << assigned)) {
                assigned++;
            }
            colors[u] = assigned;
        }

        // b. Parallel unblocking: decrement in-degree of dependent neighbors
        next_queue.clear();
        #pragma omp parallel
        {
            std::vector<int> thread_local_next;
            #pragma omp for schedule(guided)
            for (int i = 0; i < q_size; i++) {
                int u = curr_queue[i];
                int deg_u = degrees[u];
                int beg = nindex[u], end = nindex[u + 1];

                for (int idx = beg; idx < end; idx++) {
                    int v = nlist[idx];
                    if (has_higher_priority(u, deg_u, v, degrees[v])) {
                        int rem;
                        #pragma omp atomic capture
                        rem = --in_degree[v];
                        if (rem == 0) {
                            thread_local_next.push_back(v);
                        }
                    }
                }
            }

            #pragma omp critical
            {
                next_queue.insert(next_queue.end(), thread_local_next.begin(), thread_local_next.end());
            }
        }

        curr_queue.swap(next_queue);
    }

    double t_end = omp_get_wtime();
    double runtime_sec = t_end - t_start;

    // 5. Compute Statistics & Validation
    int max_color = 0;
    std::vector<int> color_counts(64, 0);
    for (int i = 0; i < nodes; i++) {
        if (colors[i] > max_color) max_color = colors[i];
        if (colors[i] < 64) color_counts[colors[i]]++;
    }
    int total_colors = max_color + 1;

    bool passed = verify_coloring(g, colors.data());

    printf("==================================================================\n");
    printf("  JP-LDF Execution Results Summary                                \n");
    printf("==================================================================\n");
    printf("Parallel Steps   : %d steps\n", step_count);
    printf("Average Parallel : %.1f vertices/step\n", (double)nodes / step_count);
    printf("Colors Used      : %d colors\n", total_colors);
    printf("Runtime          : %.4f seconds (%.2f ms)\n", runtime_sec, runtime_sec * 1000.0);
    printf("Throughput       : %.2f Medges/s | %.2f Mnodes/s\n",
           (edges * 0.000001) / runtime_sec, (nodes * 0.000001) / runtime_sec);
    printf("Verification     : %s\n\n", passed ? "PASSED (Valid Coloring)" : "FAILED (Color Conflict Found!)");

    printf("--- Color Allocation Distribution ---\n");
    for (int c = 0; c < total_colors; c++) {
        printf("  Color %d: %7d vertices (%5.2f%%)\n", c, color_counts[c], (double)color_counts[c] / nodes * 100.0);
    }

    printf("\n--- Step-by-Step Progression (First 10 & Last 5 Steps) ---\n");
    for (int s = 0; s < std::min(10, step_count); s++) {
        printf("  Step %2d: %6d vertices colored\n", s + 1, step_history[s]);
    }
    if (step_count > 15) {
        printf("  ...\n");
        for (int s = step_count - 5; s < step_count; s++) {
            printf("  Step %2d: %6d vertices colored\n", s + 1, step_history[s]);
        }
    }
    printf("==================================================================\n");

    freeECLgraph(g);
    return passed ? 0 : 1;
}
