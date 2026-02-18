/*
 * Graph-tool C++ extension module for reverse-greedy reinsertion.
 *
 * This module operates directly on graph_tool::GraphInterface, eliminating
 * the need for temp-file I/O and subprocess calls.  The algorithm is
 * identical to reinsertion.cpp but works in-process with the graph-tool
 * graph object.
 *
 * Algorithm (reverse-greedy reinsertion):
 *   1. Start with all non-seed vertices present; build union-find.
 *   2. Greedily reinsert the seed vertex whose reinsertion increases
 *      the largest connected component (LCC) the least.
 *   3. Stop when the LCC reaches the target size.
 *   4. Return the remaining seed vertices (optimized dismantling set),
 *      optionally sorted by degree.
 *
 * Important: graph-tool's internal adj_list is always directed.  For
 * undirected graphs, edges are stored in one direction only.  We build
 * a symmetric adjacency list at startup to handle both directions,
 * matching the behaviour of the subprocess-based reinsertion binary.
 *
 * Based on the reverse-greedy idea from reverse-greedy.cpp by
 * Alfredo Braunstein (https://github.com/abraunst/decycler).
 *
 * Build: make libreinsertion_gt.so  (see Makefile in this directory)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2.
 */

#include <graph.hh>
#include <graph_python_interface.hh>

#include <boost/python.hpp>
#include <boost/pending/disjoint_sets.hpp>

#include <algorithm>
#include <utility>
#include <vector>

namespace gt = graph_tool;
namespace py = boost::python;

using multigraph_t = gt::GraphInterface::multigraph_t;
using vertex_t = boost::graph_traits<multigraph_t>::vertex_descriptor;

// -------------------------------------------------------------------------
// Core algorithm helpers
// -------------------------------------------------------------------------

/**
 * Compute the connected-component size that vertex @p i would belong to
 * if it were reinserted into the network.
 *
 * @param i           Vertex to hypothetically reinsert.
 * @param adj         Symmetric adjacency list (both edge directions).
 * @param present     present[v] == 1 iff v is currently in the network.
 * @param size_comp   size_comp[root] == size of the component rooted at root.
 * @param ds          Disjoint-sets (union-find) data structure.
 * @param mask        Scratch space (size N, all zeros on entry and exit).
 * @return (component_size_after_reinsertion, number_of_distinct_neighbour_components).
 */
static std::pair<size_t, unsigned> compute_comp(
        size_t i,
        std::vector<std::vector<size_t>> const &adj,
        std::vector<int> const &present,
        std::vector<size_t> const &size_comp,
        boost::disjoint_sets<size_t *, size_t *> &ds,
        std::vector<unsigned> &mask)
{
    std::vector<size_t> compos;
    size_t nc = 1;
    unsigned ncomp = 0;

    for (size_t j : adj[i]) {
        if (present[j]) {
            size_t c = ds.find_set(j);
            if (!mask[c]) {
                compos.push_back(c);
                mask[c] = 1;
                nc += size_comp[c];
                ncomp++;
            }
        }
    }
    for (auto c : compos)
        mask[c] = 0;

    return {nc, ncomp};
}

// -------------------------------------------------------------------------
// Main reinsertion function
// -------------------------------------------------------------------------

/**
 * Run reverse-greedy reinsertion directly on a graph-tool Graph.
 *
 * @param py_graph       graph_tool.Graph Python object.
 * @param py_seeds       Python list of vertex indices (0-based) to remove.
 * @param target_size    Stop reinserting when LCC >= target_size.
 * @param sort_strategy  0 = original order, 1 = ascending degree, 2 = descending.
 * @return Python list of remaining seed vertex indices after optimization.
 */
static py::list reverse_greedy_impl(
        py::object py_graph,
        py::list py_seeds,
        unsigned target_size,
        int sort_strategy)
{
    // Extract internal GraphInterface from graph_tool.Graph Python object
    py::object gi_obj = py_graph.attr("_Graph__graph");
    gt::GraphInterface &gi = py::extract<gt::GraphInterface &>(gi_obj);
    multigraph_t const &g = gi.get_graph();
    size_t const N = num_vertices(g);
    bool const directed = gi.get_directed();

    if (N == 0)
        return py::list();

    // ---- Build symmetric adjacency list ----
    // graph-tool's adj_list stores edges in one direction only for
    // undirected graphs.  We must add both directions explicitly,
    // matching the behaviour of the subprocess binary (which reads
    // an edge-list file into a BGL undirectedS graph).
    std::vector<std::vector<size_t>> adj(N);

    for (auto [vi, vi_end] = vertices(g); vi != vi_end; ++vi) {
        size_t v = *vi;
        auto [adj_begin, adj_end] = adjacent_vertices(v, g);
        for (auto it = adj_begin; it != adj_end; ++it) {
            size_t u = *it;
            adj[v].push_back(u);
            if (!directed)
                adj[u].push_back(v);
        }
    }

    // Deduplicate neighbours (handles multi-edges and both-direction storage)
    for (auto &nbrs : adj) {
        std::sort(nbrs.begin(), nbrs.end());
        nbrs.erase(std::unique(nbrs.begin(), nbrs.end()), nbrs.end());
    }

    // ---- Build seed set from Python list ----
    std::vector<int> seed(N, 0);
    int nseed = 0;
    py::ssize_t const num_py_seeds = py::len(py_seeds);
    for (py::ssize_t i = 0; i < num_py_seeds; ++i) {
        size_t v = py::extract<size_t>(py_seeds[i]);
        if (v < N && !seed[v]) {
            seed[v] = 1;
            nseed++;
        }
    }

    if (nseed <= 1) {
        // Nothing to optimise — return all seeds as-is
        py::list result;
        for (size_t i = 0; i < N; ++i)
            if (seed[i])
                result.append(i);
        return result;
    }

    // ---- Union-find data structures ----
    std::vector<size_t> rank_vec(N, 0);
    std::vector<size_t> parent_vec(N);
    boost::disjoint_sets<size_t *, size_t *> ds(&rank_vec[0], &parent_vec[0]);

    std::vector<int> present(N, 0);
    std::vector<size_t> size_comp(N, 0);
    std::vector<unsigned> mask(N, 0);

    for (size_t i = 0; i < N; ++i)
        ds.make_set(i);

    size_t ngiant = 0;

    // ---- Phase 1: insert all non-seed vertices, build union-find ----
    for (size_t i = 0; i < N; ++i) {
        if (seed[i])
            continue;

        auto [nc, ncomp] = compute_comp(i, adj, present, size_comp, ds, mask);
        (void)ncomp;
        present[i] = 1;

        for (size_t j : adj[i]) {
            if (present[j])
                ds.union_set(i, j);
        }
        size_comp[ds.find_set(i)] = nc;
        if (nc > ngiant)
            ngiant = nc;
    }

    // ---- Phase 2: greedy reinsertion of seed nodes ----
    for (int t = nseed; --t > 0; ) {
        size_t nbest = N + 1;
        size_t ibest = 0;

        for (size_t i = 0; i < N; ++i) {
            if (present[i])
                continue;
            auto [nc, ncomp] = compute_comp(i, adj, present, size_comp, ds, mask);
            (void)ncomp;
            if (nc < nbest) {
                ibest = i;
                nbest = nc;
            }
        }

        present[ibest] = 1;
        for (size_t j : adj[ibest]) {
            if (present[j])
                ds.union_set(ibest, j);
        }
        size_comp[ds.find_set(ibest)] = nbest;

        if (nbest > ngiant)
            ngiant = nbest;
        if (nbest >= target_size)
            break;
        seed[ibest] = 0;
    }

    // ---- Collect remaining seeds ----
    std::vector<size_t> remaining;
    remaining.reserve(nseed);
    for (size_t i = 0; i < N; ++i) {
        if (seed[i])
            remaining.push_back(i);
    }

    // ---- Sort by degree if requested ----
    if (sort_strategy == 1) {
        // Ascending degree (smaller first)
        std::sort(remaining.begin(), remaining.end(),
                  [&adj](size_t a, size_t b) {
                      return adj[a].size() < adj[b].size();
                  });
    } else if (sort_strategy == 2) {
        // Descending degree (larger first)
        std::sort(remaining.begin(), remaining.end(),
                  [&adj](size_t a, size_t b) {
                      return adj[a].size() > adj[b].size();
                  });
    }

    // Convert to Python list
    py::list result;
    for (size_t v : remaining)
        result.append(v);
    return result;
}

// -------------------------------------------------------------------------
// Boost.Python module definition
// -------------------------------------------------------------------------
BOOST_PYTHON_MODULE(libreinsertion_gt)
{
    py::def("reverse_greedy_reinsertion", &reverse_greedy_impl,
            (py::arg("graph"),
             py::arg("seeds"),
             py::arg("target_size"),
             py::arg("sort_strategy") = 2),
            "Run reverse-greedy reinsertion directly on a graph-tool Graph.\n\n"
            "Args:\n"
            "    graph: graph_tool.Graph (undirected or directed)\n"
            "    seeds: list of vertex indices (0-based) to consider for removal\n"
            "    target_size: stop reinserting when LCC >= target_size\n"
            "    sort_strategy: 0=original, 1=ascending degree, 2=descending degree\n\n"
            "Returns:\n"
            "    list of remaining seed vertex indices after reinsertion,\n"
            "    sorted by degree according to sort_strategy"
    );
}
