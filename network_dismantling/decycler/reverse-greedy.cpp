/*
 * A reverse-greedy procedure for the dismantling problem
 * Copyright (C) 2016 Alfredo Braunstein
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



/*
 * This code reinserts removed nodes (S vertices) in a greedy way.
 * The starting graph (once S vertices have been removed) must be acyclic.
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

#include "real_type.hpp"

using namespace boost;
using namespace std;

namespace params {

real_t threshold = 1000000;

}

using namespace params;



typedef adjacency_list<vecS, vecS, undirectedS> Graph;
typedef graph_traits<Graph>::out_edge_iterator edge_iterator;
typedef graph_traits<Graph>::edge_descriptor Edge;
typedef graph_traits<Graph>::vertex_descriptor Vertex;
typedef graph_traits<Graph>::vertices_size_type VertexIndex;
typedef VertexIndex* Rank;
typedef Vertex* Parent;


Graph g;

unsigned N = 0;
vector<int> seed;
int nseed = 0;

void read_graph(istream & file)
{
	string line;

	while (getline(file, line)) {
		istringstream iline(line.c_str());
		string tok, tok2;
		iline >> tok;
		if (tok == "V" || tok == "#" ) {
			continue;
		} else if (tok == "S") {
			int i;
			iline >> i;
			seed.resize(max(i + 1, int(seed.size())));
			seed[i] = 1;
			nseed++;
		} else if (tok == "D" || tok == "E") {
			int i, j;
			iline >> i >> j;
			add_edge(i, j, g);
		} else {
			cout << "token " << tok << " unknown" << endl;
			assert(0);
		}
	}

	N = num_vertices(g);
	seed.resize(N);
	cerr << num_edges(g) << " edges, " << N << " vertices" << endl;
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


void run_greedy()
{
	vector<VertexIndex> rank(N);
	vector<Vertex> parent(N);
	vector<int> handle(N);
	vector<int> present(N);
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
	for (unsigned t = nseed; --t; ) {
		long nbest = N;
		unsigned ibest = 0;
		int ncompbest = 0;
		for (unsigned i = 0; i < N; ++i) {
			if (present[i])
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
		if (t % 1 == 0)
			cout << t << " " << ngiant << " " << ibest << " " << nbest << " " << num_comp << " " << nedges <<  endl;
	}
	for (unsigned i = 0; i < N; ++i) {
		if (seed[i])
			cout << "S " << i << endl;
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
	desc.add_options()
		("help,h", "produce help message")
		("threshold,t", po::value(&threshold)->default_value(threshold), 
		 "stop on threshold");

	po::variables_map vm;
	po::store(po::parse_command_line(ac, av, desc), vm);
	po::notify(vm);

	if (vm.count("help")) {
		cerr << desc << "\n";
		exit(1);
	}

	return vm;
}


int main(int ac, char** av)
{
	po::variables_map vm = parse_command_line(ac, av);
	read_graph(cin);
	run_greedy();
	return 0;
}


