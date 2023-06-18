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

#include <boost/config.hpp>
#include <boost/static_assert.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/breadth_first_search.hpp>
#include <boost/graph/visitors.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/random.hpp>
#include <boost/random/linear_congruential.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/graph/erdos_renyi_generator.hpp>
#include <boost/program_options.hpp>
#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/incremental_components.hpp>
#include <boost/pending/disjoint_sets.hpp>
#include <boost/heap/fibonacci_heap.hpp>


#include <fstream>
#include <functional>
#include <vector>
#include <utility>
#include <string>
#include <math.h>
#include <iomanip>
#include <boost/limits.hpp>
#include <queue>
#include <algorithm>

using namespace boost;
using namespace std;

namespace params {

int threshold = 1000000;

}

using namespace params;

typedef adjacency_list<vecS, vecS, undirectedS> Graph;
typedef graph_traits<Graph>::out_edge_iterator edge_iterator;
typedef graph_traits<Graph>::edge_descriptor Edge;
typedef graph_traits<Graph>::vertex_descriptor Vertex;
typedef graph_traits<Graph>::vertices_size_type VertexIndex;
typedef VertexIndex* Rank;
typedef Vertex* Parent;


#include "config_r.h"


Graph g;
unsigned N = 0;
vector<int> seed;
int nseed = 0;

void read_graph() 
{
	ifstream rd(FILE_NET), rd2(FILE_ID);
	if (!rd || !rd2) std::cout << "error opening file\n";

	int id1 = 0, id2 = 0;
	while (rd >> id1 >> id2) {
		add_edge(id1, id2, g);
	}
	rd.close();

	while (rd2 >> id1) {
		seed.resize(max(id1 + 1, int(seed.size())));
		seed[id1] = 1; // here is not id1-1
		nseed++;
	}
	rd2.close();

	N = num_vertices(g);
	seed.resize(N);
	std::cout << num_edges(g) << " edges, " << N << " vertices" << endl;
}

pair<long,int> compute_comp(unsigned i, vector<int> const & present,
		vector<int> const & size_comp, disjoint_sets<Rank, Parent> & ds)
{
	static vector<int> mask(N);

	vector<int> compos;
	edge_iterator eit, eend;
	long nc = 1;
	int ncomp = 0;
	for (tie(eit, eend) = out_edges(i, g); eit != eend; ++eit) {
		int j =  target(*eit, g);
		if (present[j]) {
			int c = ds.find_set(j);
			if (!mask[c]) {
				compos.push_back(c);
				mask[c] = 1;
				nc += size_comp[c];
				ncomp++;
			}
		}
	}
	for (unsigned k = 0; k < compos.size(); ++k)
		mask[compos[k]] = 0;
	return make_pair(nc, ncomp);
}

void run_greedy(vector<int>& nodes)
{
	vector<VertexIndex> rank(N);
	vector<Vertex> parent(N);
	vector<int> handle(N);
	vector<int> present(N); // flag: the node in the network or not
	vector<int> size_comp(N);
	disjoint_sets<Rank, Parent> ds(&rank[0], &parent[0]);
	int ngiant = 0;
	for (unsigned i  = 0; i < N; ++i)
		ds.make_set(i);
	edge_iterator eit, eend;
	int num_comp = N;
	int nedges = 0;
	for (unsigned i = 0; i < N; ++i) {
		if (seed[i])
			continue;
		long nc;
		int ncomp;
		tie(nc, ncomp) = compute_comp(i, present, size_comp, ds);
		present[i] = 1;
		num_comp += 1 - ncomp;
		for (tie(eit, eend) = out_edges(i, g); eit != eend; ++eit) {
			unsigned j =  target(*eit, g);
			if (present[j]) {
				ds.union_set(i, j);
				nedges++;
			}
		}
		size_comp[ds.find_set(i)] = nc;
		if (nc > ngiant)
			ngiant = nc;
	}

	vector<int> compos;
	for (unsigned t = nseed; --t; ) { // not a seed?
		long nbest = N;  // the new size after this reinsertion? see line 212
		unsigned ibest = 0;
		int ncompbest = 0;
		for (unsigned i = 0; i < N; ++i) {
			if (present[i]) // node i is in the network
				continue;
			long nc;
			int ncomp;
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
			unsigned j =  target(*eit, g);
			if (present[j]) {
				ds.union_set(ibest, j);
				nedges++;
			}
		}
		size_comp[ds.find_set(ibest)] = nbest;

		if (nbest > ngiant)
			ngiant = nbest;
		if (nbest >= threshold)
			break;
		seed[ibest] = 0;
		//if (t % 1 == 0)
		//	cout << t << " " << ngiant << " " << ibest << " " << nbest << " " << num_comp << " " << nedges <<  endl;
	}
	for (unsigned i = 0; i < N; ++i) {
		if (seed[i]){
			nodes.push_back(i);  // here is not i+1
			// cout << i << endl;
		    // cout << "S " << i << endl; // here is not i+1
		}
	}
}

namespace po = boost::program_options;

po::variables_map parse_command_line(int ac, char ** av)
{
	po::options_description desc(
			"Implements reverse greedy from a decycled graph\n"
			"Standard input: edges (D i j)  + seeds (removed nodes, S i)\n"
			"Usage: " + string(av[0]) + " <option> ... \n\twhere <option> is one or more of"
			);
	desc.add_options()("help,h", "produce help message");

    desc.add_options()("threshold,t", po::value(&threshold)->default_value(threshold),  // threshold
		 "stop on threshold");

    desc.add_options()("network,n", po::value(netfile),  // threshold
                       "stop on threshold");

	po::variables_map vm;
	po::store(po::parse_command_line(ac, av, desc), vm);
	po::notify(vm);

	if (vm.count("help")) {
		std::cout << desc << "\n";
		exit(1);
	}

	return vm;
}

// sort the nodes according to their weights
vector<int> sort_nodes_Weights(vector<double> W, vector<int> nodes) {
	if (Sort_Strategy == 0) {  // 0: keep the original order; 1: ascending; 2: descending 
		return nodes;
	}

	vector<int> newlist;
	int target = 0; // the target node in 'nodes'
	for (int i = 0; i < int(nodes.size()); i++) { // set the target as the first removed node
		if (nodes[i] != 0) {
			target = i;
			break;
		}
	}

	if (Sort_Strategy == 1) { // 1: ascending
		while (newlist.size() != nodes.size()) {
			for (int i = 0; i < int(nodes.size()); i++) {
				if (nodes[i] != 0 && W[nodes[target] - 1] > W[nodes[i] - 1]) { // select the node with smaller degree
					target = i;
				}
			}
			newlist.push_back(nodes[target]);
			nodes[target] = 0;

			for (int i = 0; i < int(nodes.size()); i++) { // set the target as the first removed node
				if (nodes[i] != 0) {
					target = i;
					break;
				}
			}
		}
	}
	else if (Sort_Strategy == 2) { // 2: descending 
		while (newlist.size() != nodes.size()) {
			for (int i = 0; i < int(nodes.size()); i++) {
				if (nodes[i] != 0 && W[nodes[target] - 1] < W[nodes[i] - 1]) { // select the node with larger degree
					target = i;
				}
			}
			newlist.push_back(nodes[target]);
			nodes[target] = 0;

			for (int i = 0; i < int(nodes.size()); i++) { // set the target as the first removed node
				if (nodes[i] != 0) {
					target = i;
					break;
				}
			}
		}
	}

	return newlist;
}

void write(vector<int> nodes_id) {
	ofstream wt2(FILE_ID2);
	if (!wt2) std::cout << "error creating file...\n";

	if (Sort_Strategy != 0) {
		for (int i = 0; i<int(nodes_id.size()); i++)
//		std::cout << nodes_id[i] << endl;
			wt2 << nodes_id[i] << endl;
		wt2.close();
	}
}

int main(int ac, char** av)
{
	po::variables_map vm = parse_command_line(ac, av);
	read_graph();

	vector<int> nodes, nodes_ordered; // store the nodes that should be removed after reinsertion
	run_greedy(nodes); // reinsertion

	vector<double> Weights(int(N),0); // store the weights of each node

	for (int i = 0; i < int(N); i++){
		Weights[i] = degree(i+1, g); // the latter one is i+1
	}

	nodes_ordered = sort_nodes_Weights(Weights, nodes); // sort the nodes in the set nodes
	write(nodes_ordered);

	return 0;
}
