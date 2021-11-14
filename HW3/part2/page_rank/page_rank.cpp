#include "page_rank.h"

#include <omp.h>
#include <stdlib.h>

#include <cmath>
#include <utility>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is
// num_nodes(g)) damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double *solution, double damping, double convergence) {
  // initialize vertex weights to uniform probability. Double
  // precision scores are used to avoid underflow for large graphs

  int numNodes = num_nodes(g);
  double *score_old = new double[numNodes];
  double equal_prob = 1.0 / numNodes;
  for (int i = 0; i < numNodes; ++i) {
    solution[i] = equal_prob;
    score_old[i] = equal_prob;
  }

  /*
     For PP students: Implement the page rank algorithm here.  You
     are expected to parallelize the algorithm using openMP.  Your
     solution may need to allocate (and free) temporary arrays.

     Basic page rank pseudocode is provided below to get you started:

     // initialization: see example code above
     score_old[vi] = 1/numNodes;
  */

  bool converged = false;
  while (!converged) {
    double no_out = 0.0;
    double global_diff = 0.0;

#pragma omp parallel
    {
#pragma omp for reduction(+ : no_out)
      for (int i = 0; i < numNodes; ++i) {
        int o = outgoing_size(g, i);
        if (o == 0) {
          no_out += damping * score_old[i] / numNodes;
        }
      }
#pragma omp for
      for (int i = 0; i < numNodes; ++i) {
        const Vertex *begin = incoming_begin(g, i);
        const Vertex *end = incoming_end(g, i);
        double ns = 0.0;
        for (const Vertex *v = begin; v != end; ++v) {
          int edges = outgoing_size(g, *v);
          ns += score_old[*v] / edges;
        }

        solution[i] = damping * ns + (1.0 - damping) / numNodes + no_out;
      }

// compute score_new[vi] for all nodes vi:
// score_new[vi] =
//     sum over all nodes vj reachable from incoming edges{
//         score_old[vj] / number of edges leaving vj}
// score_new[vi] =
//         (damping * score_new[vi]) + (1.0 - damping) / numNodes;

// score_new[vi] +=
//     sum over all nodes v in graph with no outgoing edges{
//         damping * score_old[v] / numNodes}

// compute how much per-node scores have changed
// quit once algorithm has converged
#pragma omp for reduction(+ : global_diff)
      for (int i = 0; i < numNodes; ++i) {
        global_diff += abs(solution[i] - score_old[i]);
        score_old[i] = solution[i];
      }
#pragma omp single
      { converged = (global_diff < convergence); }
    }
  }

  delete[] score_old;
}
