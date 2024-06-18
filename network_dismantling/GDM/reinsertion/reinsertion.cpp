/*
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

/*
 * This code reinserts the removed nodes (S vertices) in a greedy way.
 * This code is a modification of the https://github.com/abraunst/decycler/blob/master/reverse-greedy.cpp 
 * which is created by Alfredo Braunstein.
 * */

#include <boost/graph/adjacency_list.hpp>
#include <boost/random/linear_congruential.hpp>
#include <boost/graph/erdos_renyi_generator.hpp>
#include <boost/program_options.hpp>
#include <boost/pending/disjoint_sets.hpp>


#include <fstream>
#include <vector>
#include <utility>
#include <string>
#include <cmath>
#include <algorithm>


using namespace boost;
using namespace std;

namespace po = boost::program_options;

namespace params {
    string FILE_NET;                 // input format of each line: id1 id2
    string FILE_ID;                  // output the id of the removed nodes in order
    string FILE_OUTPUT;                  // output the id of the removed nodes in order
    unsigned TARGET_SIZE;                // If the gcc size is smaller than TARGET_SIZE, the dismantling will stop. Default value can be 0.01*NODE_NUM  OR  1000
    int SORT_STRATEGY;                   // removing order
}

using namespace params;

typedef adjacency_list<vecS, vecS, undirectedS> Graph;
typedef graph_traits<Graph>::out_edge_iterator edge_iterator;
typedef graph_traits<Graph>::edge_descriptor Edge;
typedef graph_traits<Graph>::vertex_descriptor Vertex;
typedef graph_traits<Graph>::vertices_size_type VertexIndex;
typedef VertexIndex *Rank;
typedef Vertex *Parent;


//                            const char* fileNet = \"{}\";  // input format of each line: id1 id2\n"
//            #              "const char* fileId = \"{}\";   // output the id of the removed nodes in order\n"
//            #              "const char* outputFile = \"{}\";   // output the id of the removed nodes after reinserting\n"
//            #              "const int strategy = {}; // removing order\n"

Graph g;
unsigned N = 0;
vector<unsigned> seed;
int nseed = 0;

void read_graph() {
    ifstream rd(FILE_NET.c_str());
    if (!rd) cout << "Error opening network file\n";

    ifstream rd2(FILE_ID.c_str());
    if (!rd2) cout << "Error opening ID file\n";

    unsigned id1 = 0, id2 = 0;
    while (rd >> id1 >> id2) {
        add_edge(id1, id2, g);
    }
    rd.close();

    while (rd2 >> id1) {
        seed.resize(max(id1 + 1, unsigned(seed.size())));
        seed[id1] = 1; // here is not id1-1
        nseed++;
    }
    rd2.close();

    N = num_vertices(g);
    seed.resize(N);
    cout << num_edges(g) << " edges, " << N << " vertices" << endl;
    cout << "Seed size: " << nseed << endl;
}

pair<long, unsigned> compute_comp(unsigned i, vector<int> const &present,
                                  vector<unsigned> const &size_comp, disjoint_sets<Rank, Parent> &ds) {
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
    for (unsigned compo: compos)
        mask[compo] = 0;

    return make_pair(nc, ncomp);
}

void run_greedy(vector<unsigned> &nodes) {
    vector<VertexIndex> rank(N);
    vector<Vertex> parent(N);
    vector<int> handle(N);
    vector<int> present(N); // flag: the node in the network or not
    vector<unsigned> size_comp(N);
    disjoint_sets<Rank, Parent> ds(&rank[0], &parent[0]);

    unsigned long ngiant = 0;
    for (unsigned i = 0; i < N; ++i)
        ds.make_set(i);

    edge_iterator eit, eend;
    unsigned long num_comp = N;
    unsigned nedges = 0;

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

    vector<unsigned> compos;
    for (unsigned t = nseed; --t;) { // not a seed?
        unsigned long nbest = N;  // the new size after this reinsertion? see line 212
        unsigned ibest = 0;
        unsigned ncompbest = 0;
        for (unsigned i = 0; i < N; ++i) {
            if (present[i]) // node i is in the network
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
        //if (t % 1 == 0)
        //	cout << t << " " << ngiant << " " << ibest << " " << nbest << " " << num_comp << " " << nedges <<  endl;
    }
    for (unsigned i = 0; i < N; ++i) {
        if (seed[i]) {
            nodes.push_back(i);  // here is not i+1
            // cout << i << endl;
            // cout << "S " << i << endl; // here is not i+1
        }
    }
}

namespace po = boost::program_options;

po::variables_map parse_command_line(int ac, char **av) {
    po::options_description desc(
            "Implements reverse greedy from a decycled graph\n"
            "Standard input: edges (D i j)  + seeds (removed nodes, S i)\n"
            "Usage: " + string(av[0]) + " <option> ... \n\twhere <option> is one or more of"
    );
    desc.add_options()
            ("help", "produce help message")
            ("NetworkFile,NF", po::value(&FILE_NET), "File containing the network")
            ("IDFile,IF", po::value(&FILE_ID), "File containing the network")
            ("OutFile,OF", po::value(&FILE_OUTPUT), "File containing the network")
            ("TargetSize,t", po::value(&TARGET_SIZE), "Target component size (int)")
            ("SortStrategy,S", po::value(&SORT_STRATEGY),
             "Removal strategy (0: keep the original order; 1: ascending; 2: descending)");

    po::variables_map vm;
    po::store(po::parse_command_line(ac, av, desc), vm);
    po::notify(vm);

    if (!vm.count("NetworkFile") || !vm.count("IDFile") || !vm.count("OutFile") || !vm.count("TargetSize") ||
        !vm.count("SortStrategy")) {
        cout << desc << "\n";
        exit(1);
    }

    printf("NetworkFile: %s\n", FILE_NET.c_str());
    printf("IDFile: %s\n", FILE_ID.c_str());
    printf("OutFile: %s\n", FILE_OUTPUT.c_str());
    printf("TargetSize: %d\n", TARGET_SIZE);
    printf("SortStrategy: %d\n", SORT_STRATEGY);

    if (vm.count("help")) {
        cout << desc << "\n";
        exit(1);
    }

    return vm;
}

// sort the nodes according to their weights
vector<unsigned> sort_nodes_Weights(vector<unsigned> W, vector<unsigned> nodes) {
    // 0: keep the original order; 1: ascending; 2: descending
    if (SORT_STRATEGY == 0) {
        return nodes;
    }

    vector<unsigned> newlist;
    unsigned target = 0; // the target node in 'nodes'
    for (unsigned i = 0; i < nodes.size(); i++) { // set the target as the first removed node
        if (nodes[i] != 0) {
            target = i;
            break;
        }
    }

    if (SORT_STRATEGY == 1) { // 1: ascending
        while (newlist.size() != nodes.size()) {
            for (unsigned i = 0; i < nodes.size(); i++) {
                if (nodes[i] != 0 && W[nodes[target] - 1] > W[nodes[i] - 1]) { // select the node with smaller degree
                    target = i;
                }
            }
            newlist.push_back(nodes[target]);
            nodes[target] = 0;

            for (unsigned i = 0; i < nodes.size(); i++) { // set the target as the first removed node
                if (nodes[i] != 0) {
                    target = i;
                    break;
                }
            }
        }
    } else if (SORT_STRATEGY == 2) { // 2: descending
        while (newlist.size() != nodes.size()) {
            for (unsigned i = 0; i < nodes.size(); i++) {
                if (nodes[i] != 0 && W[nodes[target] - 1] < W[nodes[i] - 1]) { // select the node with larger degree
                    target = i;
                }
            }
            newlist.push_back(nodes[target]);
            nodes[target] = 0;

            for (unsigned i = 0; i < nodes.size(); i++) { // set the target as the first removed node
                if (nodes[i] != 0) {
                    target = i;
                    break;
                }
            }
        }
    }

    return newlist;
}

void write(const vector<unsigned> &nodes_id) {
    ofstream wt2(FILE_OUTPUT.c_str());
    if (!wt2) {
        cout << "error creating file...\n";
        exit(-1);
    }

    if (SORT_STRATEGY != 0) {
        for (unsigned i: nodes_id)
//		cout << nodes_id[i] << endl;
            wt2 << i << endl;
        wt2.close();
    }
}

int main(int ac, char **av) {
    po::variables_map vm = parse_command_line(ac, av);
    read_graph();


    vector<unsigned> nodes, nodes_ordered; // store the nodes that should be removed after reinsertion

    cout << "Running greedy reinsertion algorithm..." << endl;
    run_greedy(nodes); // reinsertion

//    cout << "Greedy output:" << endl;
//    for (unsigned i: nodes) {
//        cout << i << endl;
//    }

    // store the weights of each node
    vector<unsigned> Weights(N, 0);

    for (unsigned int i = 0; i < N; i++) {
        Weights[i] = degree(i + 1, g); // the latter one is i+1
    }

//    cout << "Degree:" << endl;
//    for (unsigned int i: nodes) {
//        cout << Weights[i] << endl;
//    }

    nodes_ordered = sort_nodes_Weights(Weights, nodes); // sort the nodes in the set nodes

//    cout << "Ordered nodes:" << endl;
//    for (unsigned int i: nodes_ordered) {
//        cout << i << endl;
//    }
    write(nodes_ordered);

    return 0;
}
