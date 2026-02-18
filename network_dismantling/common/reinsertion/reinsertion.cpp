/*
 * Reverse-greedy reinsertion algorithm for network dismantling.
 *
 * Given a set of "seed" (removed) nodes, this algorithm greedily reinserts
 * seed nodes back into the network — always picking the one whose reinsertion
 * increases the largest connected component (LCC) the least — until the LCC
 * reaches the target size.  The remaining seed nodes form the optimised
 * dismantling set.
 *
 * Based on the reverse-greedy idea from:
 *   https://github.com/abraunst/decycler/blob/master/reverse-greedy.cpp
 *   (Alfredo Braunstein)
 *
 * Dependencies: Boost Graph Library, Boost.ProgramOptions
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation version 2 of the License.
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include <boost/graph/adjacency_list.hpp>
#include <boost/random/linear_congruential.hpp>
#include <boost/graph/erdos_renyi_generator.hpp>
#include <boost/program_options.hpp>
#include <boost/pending/disjoint_sets.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

using namespace boost;
using namespace std;

namespace po = boost::program_options;

// ---------------------------------------------------------------------------
// Parameters (populated from command line)
// ---------------------------------------------------------------------------
namespace params {
    string FILE_NET;       // edge-list file
    string FILE_ID;        // file listing seed (removed) node IDs
    string FILE_OUTPUT;    // output file: ordered dismantling set
    unsigned TARGET_SIZE;  // stop reinserting when LCC >= TARGET_SIZE
    int SORT_STRATEGY;     // 0 = original order, 1 = ascending degree, 2 = descending degree
}

using namespace params;

// ---------------------------------------------------------------------------
// Graph types (Boost.Graph)
// ---------------------------------------------------------------------------
typedef adjacency_list<vecS, vecS, undirectedS> Graph;
typedef graph_traits<Graph>::out_edge_iterator edge_iterator;
typedef graph_traits<Graph>::edge_descriptor Edge;
typedef graph_traits<Graph>::vertex_descriptor Vertex;
typedef graph_traits<Graph>::vertices_size_type VertexIndex;
typedef VertexIndex *Rank;
typedef Vertex *Parent;

// ---------------------------------------------------------------------------
// Globals
// ---------------------------------------------------------------------------
Graph g;
unsigned N = 0;
vector<unsigned> seed;
int nseed = 0;

// ---------------------------------------------------------------------------
// I/O
// ---------------------------------------------------------------------------
void read_graph() {
    ifstream rd(FILE_NET.c_str());
    if (!rd) {
        cerr << "Error opening network file: " << FILE_NET << endl;
        exit(1);
    }

    ifstream rd2(FILE_ID.c_str());
    if (!rd2) {
        cerr << "Error opening ID file: " << FILE_ID << endl;
        exit(1);
    }

    unsigned id1 = 0, id2 = 0;
    while (rd >> id1 >> id2) {
        add_edge(id1, id2, g);
    }
    rd.close();

    while (rd2 >> id1) {
        seed.resize(max(id1 + 1, unsigned(seed.size())));
        seed[id1] = 1;
        nseed++;
    }
    rd2.close();

    N = num_vertices(g);
    seed.resize(N);
    cout << num_edges(g) << " edges, " << N << " vertices" << endl;
    cout << "Seed size: " << nseed << endl;
}

// ---------------------------------------------------------------------------
// Core algorithm
// ---------------------------------------------------------------------------

/**
 * Compute the connected-component size that vertex `i` would belong to
 * if it were reinserted into the network (using union-find `ds`).
 *
 * Returns (component_size_after_reinsertion, number_of_distinct_neighbour_components).
 */
pair<long, unsigned> compute_comp(unsigned i,
                                  vector<int> const &present,
                                  vector<unsigned> const &size_comp,
                                  disjoint_sets<Rank, Parent> &ds) {
    static vector<unsigned> mask(N);

    vector<unsigned> compos;
    edge_iterator eit, eend;
    unsigned long nc = 1;
    unsigned ncomp = 0;
    for (tie(eit, eend) = out_edges(i, g); eit != eend; ++eit) {
        unsigned long j = target(*eit, g);
        if (present[j]) {
            unsigned long c = ds.find_set(j);
            if (!mask[c]) {
                compos.push_back(c);
                mask[c] = 1;
                nc += size_comp[c];
                ncomp++;
            }
        }
    }
    for (unsigned compo : compos)
        mask[compo] = 0;

    return make_pair(nc, ncomp);
}

/**
 * Greedy reinsertion: iteratively pick the seed node whose reinsertion
 * causes the smallest increase in the LCC.  Stop when the LCC reaches
 * TARGET_SIZE.  The remaining seed nodes (still marked in `seed[]`) are
 * written to `nodes`.
 */
void run_greedy(vector<unsigned> &nodes) {
    vector<VertexIndex> rank(N);
    vector<Vertex> parent(N);
    vector<int> handle(N);
    vector<int> present(N);
    vector<unsigned> size_comp(N);
    disjoint_sets<Rank, Parent> ds(&rank[0], &parent[0]);

    unsigned long ngiant = 0;
    for (unsigned i = 0; i < N; ++i)
        ds.make_set(i);

    edge_iterator eit, eend;
    unsigned long num_comp = N;
    unsigned nedges = 0;

    // Phase 1: insert all non-seed nodes and build union-find
    for (unsigned i = 0; i < N; ++i) {
        if (seed[i])
            continue;
        unsigned long nc;
        unsigned long ncomp;
        tie(nc, ncomp) = compute_comp(i, present, size_comp, ds);
        present[i] = 1;
        num_comp += 1 - ncomp;
        for (tie(eit, eend) = out_edges(i, g); eit != eend; ++eit) {
            unsigned j = target(*eit, g);
            if (present[j]) {
                ds.union_set(i, j);
                nedges++;
            }
        }
        size_comp[ds.find_set(i)] = nc;
        if (nc > ngiant)
            ngiant = nc;
    }

    // Phase 2: greedy reinsertion of seed nodes
    for (unsigned t = nseed; --t;) {
        unsigned long nbest = N;
        unsigned ibest = 0;
        unsigned ncompbest = 0;
        for (unsigned i = 0; i < N; ++i) {
            if (present[i])
                continue;
            unsigned long nc;
            unsigned ncomp;
            tie(nc, ncomp) = compute_comp(i, present, size_comp, ds);
            if (nc < nbest) {
                ibest = i;
                nbest = nc;
                ncompbest = ncomp;
            }
        }
        present[ibest] = 1;
        num_comp += 1 - ncompbest;
        for (tie(eit, eend) = out_edges(ibest, g); eit != eend; ++eit) {
            unsigned j = target(*eit, g);
            if (present[j]) {
                ds.union_set(ibest, j);
                nedges++;
            }
        }
        size_comp[ds.find_set(ibest)] = nbest;

        if (nbest > ngiant)
            ngiant = nbest;
        if (nbest >= TARGET_SIZE)
            break;
        seed[ibest] = 0;
    }

    // Collect remaining seed nodes
    for (unsigned i = 0; i < N; ++i) {
        if (seed[i]) {
            nodes.push_back(i);
        }
    }
}

// ---------------------------------------------------------------------------
// Post-processing: sort output nodes by degree
// ---------------------------------------------------------------------------
vector<unsigned> sort_nodes_by_degree(vector<unsigned> W, vector<unsigned> nodes) {
    if (SORT_STRATEGY == 0) {
        return nodes;
    }

    vector<unsigned> newlist;
    unsigned target_idx = 0;

    // Find the first non-zero entry as initial target
    for (unsigned i = 0; i < nodes.size(); i++) {
        if (nodes[i] != 0) {
            target_idx = i;
            break;
        }
    }

    if (SORT_STRATEGY == 1) {
        // Ascending order (smaller degree first)
        while (newlist.size() != nodes.size()) {
            for (unsigned i = 0; i < nodes.size(); i++) {
                if (nodes[i] != 0 && W[nodes[target_idx] - 1] > W[nodes[i] - 1]) {
                    target_idx = i;
                }
            }
            newlist.push_back(nodes[target_idx]);
            nodes[target_idx] = 0;
            for (unsigned i = 0; i < nodes.size(); i++) {
                if (nodes[i] != 0) {
                    target_idx = i;
                    break;
                }
            }
        }
    } else if (SORT_STRATEGY == 2) {
        // Descending order (larger degree first)
        while (newlist.size() != nodes.size()) {
            for (unsigned i = 0; i < nodes.size(); i++) {
                if (nodes[i] != 0 && W[nodes[target_idx] - 1] < W[nodes[i] - 1]) {
                    target_idx = i;
                }
            }
            newlist.push_back(nodes[target_idx]);
            nodes[target_idx] = 0;
            for (unsigned i = 0; i < nodes.size(); i++) {
                if (nodes[i] != 0) {
                    target_idx = i;
                    break;
                }
            }
        }
    }

    return newlist;
}

void write_output(const vector<unsigned> &nodes_id) {
    ofstream wt(FILE_OUTPUT.c_str());
    if (!wt) {
        cerr << "Error creating output file: " << FILE_OUTPUT << endl;
        exit(1);
    }

    if (SORT_STRATEGY != 0) {
        for (unsigned i : nodes_id)
            wt << i << endl;
    }
    wt.close();
}

// ---------------------------------------------------------------------------
// Command-line parsing
// ---------------------------------------------------------------------------
po::variables_map parse_command_line(int ac, char **av) {
    po::options_description desc(
        "Reverse-greedy reinsertion for network dismantling.\n"
        "Usage: " + string(av[0]) + " <option> ...\n\twhere <option> is one or more of"
    );
    desc.add_options()
        ("help", "produce help message")
        ("NetworkFile,NF", po::value(&FILE_NET), "Edge-list file")
        ("IDFile,IF", po::value(&FILE_ID), "File listing removed node IDs")
        ("OutFile,OF", po::value(&FILE_OUTPUT), "Output file for ordered dismantling set")
        ("TargetSize,t", po::value(&TARGET_SIZE), "Target LCC size (unsigned)")
        ("SortStrategy,S", po::value(&SORT_STRATEGY),
         "Output sort: 0 = original, 1 = ascending degree, 2 = descending degree");

    po::variables_map vm;
    po::store(po::parse_command_line(ac, av, desc), vm);
    po::notify(vm);

    if (vm.count("help") ||
        !vm.count("NetworkFile") || !vm.count("IDFile") ||
        !vm.count("OutFile") || !vm.count("TargetSize") ||
        !vm.count("SortStrategy")) {
        cout << desc << "\n";
        exit(1);
    }

    return vm;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int ac, char **av) {
    po::variables_map vm = parse_command_line(ac, av);
    read_graph();

    vector<unsigned> nodes, nodes_ordered;

    cout << "Running greedy reinsertion algorithm..." << endl;
    run_greedy(nodes);

    // Compute degree for sorting
    vector<unsigned> Weights(N, 0);
    for (unsigned i = 0; i < N; i++) {
        Weights[i] = degree(i + 1, g);
    }

    nodes_ordered = sort_nodes_by_degree(Weights, nodes);
    write_output(nodes_ordered);

    return 0;
}
