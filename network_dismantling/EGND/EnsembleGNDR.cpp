/*
 * This code reinserts the removed nodes (S vertices) in a greedy way.
 * This code is based on the code of the paper 'Generalized Network Dismantling', 
 * see https://github.com/renxiaolong/Generalized-Network-Dismantling
 * 
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

//#include <Windows.h>
#include <iostream>
#include <iterator>

using namespace boost;
using namespace std;

typedef adjacency_list<vecS, vecS, undirectedS> Graph;
typedef graph_traits<Graph>::out_edge_iterator edge_iterator;
typedef graph_traits<Graph>::edge_descriptor Edge;
typedef graph_traits<Graph>::vertex_descriptor Vertex;
typedef graph_traits<Graph>::vertices_size_type VertexIndex;
typedef VertexIndex* Rank;
typedef Vertex* Parent;

string FILE_NET = "digg.txt";  // input the network, format: id id (the minimal id is 1)
int threshold = 297; // ceil(double(network.size())/100.0) - the gcc of the remaining network should be smaller than threshold
bool weighted = true;
unsigned N = 0; // number of nodes in Graph g

// In boost, the id in Graoh always start from 0. So if the input network with id from 1 to N, 
// the nodes number returned by num_vertices(g) will be N+1
// Graph g: the underingly network, all the ids start from 1
// vector<int> seed stores all the nodes that should be removed
// nseed: the total number of nodes that should be removed
vector<vector<int>> read_graph(Graph& g, string file_network, string file_nodes, vector<int>& seed, int& nseed) {
	vector<vector<int>> network;
	network.reserve(threshold * 100); // reserve capacity for vector, do not affect the real size of network

	ifstream rd(file_network), rd2(file_nodes);
	if (!rd) std::cout << "error opening file 1\n";
	if (!rd2) std::cout << "error opening file 2\n";

	int id1 = 0, id2 = 0;
	while (rd >> id1 >> id2) {
		add_edge(id1, id2, g); // id starts from 1

		if (max(id1, id2) > int(network.size()))
			network.resize(max(id1, id2));
		network[id1 - 1].push_back(id2);
		network[id2 - 1].push_back(id1);
	}
	rd.close();

	while (rd2 >> id1) {
		seed.resize(max(id1 + 1, int(seed.size())));
		seed[id1] = 1; // here is not id1-1
		nseed++;
	}
	rd2.close();

	N = num_vertices(g); // number of the nodes in g 
	seed.resize(N);
	// std::cout << num_edges(g) << " edges, " << N << " nodes," << nseed<<" nodes need to be removed" << endl;

	return network;
}

// compute the number and size of the commponet when node i and its connected edges are reinserted in the network g.
pair<long, int> compute_comp(Graph& g, unsigned i, double& cost_reduction, vector<int> const present, vector<int> const size_comp, disjoint_sets<Rank, Parent> ds) {
	static vector<int> mask(N); // ??

	vector<int> compos; // set of the roots of different set
	edge_iterator eit, eend; // point to all the edge of node i 
	long node_num_i = 1;   // number of nodes in the found set that i connected to ??
	int component_num = 0; // number of the components/sets
	cost_reduction = 0;
	for (tie(eit, eend) = out_edges(i, g); eit != eend; ++eit) { // out_edges(i, g) return the iterators of the list of the end points of the out edges for node i in g
		int j = target(*eit, g); // get the pointed nodes j (edge: i->j)
		if (present[j]) {        // if node j exists
			cost_reduction += 1;
			int c = ds.find_set(j); // Finds the representative(root) of the set that i is an element of.
			if (!mask[c]) {
				compos.push_back(c);
				mask[c] = 1; // mark the representative with 1
				node_num_i += size_comp[c]; // update the number of nodes in the found set that i connected to.
				component_num++; // count the number of the found components
			}
		}
	}
	for (unsigned k = 0; k < compos.size(); ++k)
		mask[compos[k]] = 0;
	return make_pair(node_num_i, component_num);
}

// seed_final: store the id of all the nodes that should be removed after reinsertion
// seed: mark all the removed nodes with 1 before reinsertion
// seed_num: the number of nodes that should be removed
// N: number of the total nodes in g
void run_greedy(Graph& g, vector<int>& seed_final, vector<int> seed, int seed_num) { // here [Graph& g] contains & for speed up
	vector<VertexIndex> rank(N); 
	vector<Vertex> parent(N);
	vector<int> handle(N);
	vector<int> present(N); // flag: the node in the network or not
	vector<int> size_comp(N);

	// If two nodes are conected, they are in the same disjoint set.
	// There is always a single unique representative of each set. A simple rule to identify representative is, if i is the representative of a set, then parent[i] = i.
	// If i is a representative of a set, rank[i] is the height of the tree representing the set.
	// https://www.geeksforgeeks.org/disjoint-set-data-structures/
	disjoint_sets<Rank, Parent> ds(&rank[0], &parent[0]); 

	int gcc_size = 0; // number of nodes in gcc
	for (unsigned i = 0; i < N; ++i)
		ds.make_set(i);
	edge_iterator eit, eend;
	int num_comp = N; // number of sets/components in g. Initial: assume all the nodes are isolated
	int nedges = 0;

	// compute the number and size of the components of g when all the nodes in seed[] are absent.
	// Initially assuming all the nodes are isolate (no edge in g),
	// then adding node to the network (disjoint set ds) one by one.
	for (unsigned i = 0; i < N; ++i) { // for each node
		if (seed[i]) // skip i when it is in the removal set
			continue;
		long node_num_i;  // number of nodes in the component that i belongs to.
		int component_num;// number of the components in g
		double cost_reduction = 0;  // reduced cost if node i is reinserted
		tie(node_num_i, component_num) = compute_comp(g, i, cost_reduction, present, size_comp, ds); // compute the size and number of the components when i appears
		present[i] = 1; // mark i as existence
		num_comp += 1 - component_num; // update the number of components
		for (tie(eit, eend) = out_edges(i, g); eit != eend; ++eit) { // for each connected edge of i 
			unsigned j = target(*eit, g);
			if (present[j]) {
				ds.union_set(i, j); // if two end-points exist, put the edge in the disjoint set ds
				nedges++;
			}
		}
		size_comp[ds.find_set(i)] = node_num_i; // find_set: returns the Element representative of the set containing Element i and applies path compression.
		if (node_num_i > gcc_size)
			gcc_size = node_num_i;
	}

	for (unsigned t = seed_num; --t; ) { // find the best node to reinsert in each iteration
		long min_new_size = N;  // stroe the least size after every node was reinserted respectively
		unsigned index_size = -1; // mark the node i whose reinsertion will bring lest increase of gcc

		double max_cost_reduced = 0; // store the largest cost reduction after one node was reinserted
		int index_cost = -1; // mark the subscript of the node

		int ncompbest = 0;  // number of components
		for (unsigned i = 0; i < N; ++i) { // for each node
			if (present[i]) // skip if node i is existed in the network
				continue;
			long node_num_i; // number of nodes in the component that i belongs to after i is reinserted.
			int component_num; // number of the components in g
			double cost_reduction = 0; // reduced cost if node i is reinserted
			tie(node_num_i, component_num) = compute_comp(g, i, cost_reduction, present, size_comp, ds); // compute the size and number of the components if i appears
			if (!weighted && node_num_i < min_new_size) { // find and mark the node whose reinsertion will bring least increase of gcc
				index_size = i;
				min_new_size = node_num_i;
				ncompbest = component_num;
				// max_cost_reduced = cost_reduction; // doesn't have real meaning, only for outputing
			}
			if (weighted && node_num_i <= min_new_size && cost_reduction >= max_cost_reduced && node_num_i < threshold) { // find and mark the node that can greatly reduce the removal cost
				index_cost = i;
				max_cost_reduced = cost_reduction;
				ncompbest = component_num;
				min_new_size = node_num_i; // doesn't have real meaning, only for updating the state of the network
				ncompbest = component_num; // doesn't have real meaning, only for updating the state of the network
			}
		}

		// start to reinsert the node
		int index = (weighted ? index_cost : index_size);

		if (index != -1) {
			present[index] = 1;
			num_comp += 1 - ncompbest;
			for (tie(eit, eend) = out_edges(index, g); eit != eend; ++eit) { // reinsert the connected edges
				unsigned j = target(*eit, g);
				if (present[j]) {
					ds.union_set(unsigned(index), j);
					nedges++;
				}
			}
			size_comp[ds.find_set(index)] = min_new_size;

			if (!weighted && min_new_size >= threshold) // becase every time always find the minimal new size, so if it is bigger than threshold, break!
				break;
			seed[index] = 0; // remove this node from the removal set
			// cout << index + 1 << " " << max_cost_reduced << "\n";
		}
		else break; 
	}
	for (unsigned i = 0; i < N; ++i) {
		if (seed[i]) {
			seed_final.push_back(i);  // here is not i+1
		}
	}
}

namespace po = boost::program_options;

// command line: ./reinsertion -t 7 -w false
po::variables_map parse_command_line(int ac, char** av) {
	po::options_description desc(
		"Implements reverse greedy from a decycled graph\n"
		"Standard input: edges (D i j)  + seeds (removed nodes, S i)\n"
		"Usage: " + string(av[0]) + " <option> ... \n\twhere <option> is one or more of"
	);
	desc.add_options()
		("help,h", "produce help message")
		("threshold,t", po::value(&threshold)->default_value(threshold), "stop on threshold")
		("weighted,w", po::value(&weighted)->default_value(weighted), "weighted or unit case");

	po::variables_map vm;
	po::store(po::parse_command_line(ac, av, desc), vm);
	po::notify(vm);

	if (vm.count("help")) {
		std::cout << desc << "\n";
		exit(1);
	}

	return vm;
}

// sort the nodes according to their weights, for weighted or unweited case respectively
vector<int> sort_nodes_Weights(vector<double> W, vector<int> nodes) {
	//if (strategy == 0) {  // 0: keep the original order; 1: ascending; 2: descending
	//	return nodes;
	//}

	vector<int> newlist;
	int target = 0; // the target node in 'nodes'
	for (int i = 0; i < int(nodes.size()); i++) { // set the target as the first removed node
		if (nodes[i] != 0) {
			target = i;
			break;
		}
	}

	if (weighted) { // ascending for weighted case
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
	else { // descending for unweighted case
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

//void write(vector<int> nodes_id, string file_name) {
//	ofstream wt2(file_name);
//	if (!wt2) std::cout << "error creating file...\n";
//
//	if (strategy != 0) {
//		for (int i = 0; i<int(nodes_id.size()); i++)
//			wt2 << nodes_id[i] << "\n";
//		wt2.close();
//	}
//}

// contain several line for Cygwin and Visual Studio respectively
vector<string> getFileNames() {
	string FILE_PATH = "S\\";
	string  PATHS_STORE = "S\\_000.txt"; //the file to store all the file names

	//string FILE_PATH = "S/"; // for Cygwin g++
	//string  PATHS_STORE = "S/_000.txt"; //the file to store all the file names

	char szCommand[MAX_PATH] = { 0 }; 
	string command = "dir /a-d /b %s > " + PATHS_STORE; // command line for Visual Studio
	// string command = "ls -1 %s > " + PATHS_STORE; // command line for Cygwin
	wsprintfA(szCommand, command.c_str(), FILE_PATH.c_str());
	system(szCommand);

	ifstream rd(PATHS_STORE);
	if (!rd) cout << "error open file 3382! \n";

	vector<string> filepathes;
	string str = "";
	while (getline(rd, str))
		if (str != "_000.txt") filepathes.push_back(str);//去除文件名为_000的
	rd.close();

	return filepathes;
}

// return the number of nodes in the gcc
int get_gcc(vector<vector<int>> adj) {
	int n = int(adj.size());
	vector<int> cluster_id(n, 0); // store the cluster id of each node

	int id_now = 0;
	for (int i = 0; i < n; i++) // wide-first search, assign each connected cluster an id
	{
		if (cluster_id[i] == 0 && adj[i].size() > 0) { // this node does not belong to any cluster yet && this node is not isolated
			set<int> set_nodes;
			set_nodes.insert(i + 1);
			id_now++;
			while (set_nodes.size() > 0) {
				int node_now = *(--set_nodes.end());
				cluster_id[node_now - 1] = id_now;

				set_nodes.erase(--set_nodes.end());  // erase
				for (int k = 0; k<int(adj[node_now - 1].size()); k++) // append
					if (cluster_id[adj[node_now - 1][k] - 1] == 0 && adj[adj[node_now - 1][k] - 1].size() != 0)
						set_nodes.insert(adj[node_now - 1][k]);
			}
		}
	}

	int max_id = *max_element(cluster_id.begin(), cluster_id.end()); // store the max cluster_id of the connected clusters
	int gcc_size = 0;
	if (max_id != 0) {  // max_id == 0 means the network is not connected, i.e. all the nodes are isolated
		vector<int> count(max_id, 0); // count the number of nodes in the clusters
		for (int i = 0; i < n; i++)
			if (cluster_id[i] != 0)
				count[cluster_id[i] - 1]++;

		for (int i = 0; i < max_id; i++) // find the cluster with most nodes
			if (gcc_size < count[i])
				gcc_size = count[i];

	}
	return gcc_size;
}

// remove nodes from the original network and output the results
// the parameter (i.g., gcc_size, node_cost, node_number) should be empty
// return the removal cost (ratio)
double remove_write(vector<vector<int>>& network, vector<int> list, string file_name) {
	vector<double> gcc_size;
	vector<double> node_cost;
	vector<double> node_number;
	gcc_size.reserve(int(list.size()));
	node_cost.reserve(int(list.size()));
	node_number.reserve(int(list.size()));

	double total_nodes = network.size(), total_cost = 0;
	for (int i = 0; i<int(network.size()); i++)
		total_cost += network[i].size();
	total_cost = total_cost / 2;

	int removed_nodes = 0, removed_links = 0;
	for (int i = 0; i<int(list.size()); i++) { // remove node one by one
		removed_nodes++;
		removed_links += int(network[list[i] - 1].size());
		for (int j = 0; j<int(network[list[i] - 1].size()); j++) { // traversing every neighbor
			for (int k = int(network[network[list[i] - 1][j] - 1].size()); k >= 0; k--) {
				int neighbor = network[list[i] - 1][j];
				network[neighbor - 1].erase(remove(network[neighbor - 1].begin(), network[neighbor - 1].end(), list[i]), // remove node list[i] from its neighbors
					network[neighbor - 1].end());
			}
		}
		network[list[i] - 1].clear();
		gcc_size.push_back(get_gcc(network));
		node_cost.push_back(removed_links);
		node_number.push_back(removed_nodes);
	}

	// output
	ofstream write("R/GNDR_List_" + file_name), write2("R/GNDR_Plot_" + file_name);
	if (!write || !write2) cout << "Error creating output file...\n";

	for (int i = 0; i<int(list.size()); i++)
		write << list[i] << "\n";
	for (int i = 0; i<int(gcc_size.size()); i++)
		// write2 << gcc_size[i] << " " << node_cost[i] << " " << node_number[i] << "\n";
	    write2 << gcc_size[i] / total_nodes << " " << node_cost[i] / total_cost << " " << node_number[i] / total_nodes << "\n";
	write.close();
	write2.close();

	return node_cost.back() / total_cost;
}

int main(int ac, char** av) {
	po::variables_map vm = parse_command_line(ac, av);

	vector<string> file_names = getFileNames();
	vector<int> nodes_size;
	vector<double> nodes_cost;
	vector<vector<int>> seed_set; // store all the sets of the seed after reinsertion
	vector<double> Weights; // store the weight/cost of each node
	for (int nameSub = 0; nameSub<int(file_names.size()); nameSub++) {
		string file_list = "S/" + file_names[nameSub]; // path & name of the removed list
		vector<int> seed; // mark the nodes that should be removed with 1, else with 0
		int nseed = 0;    // number of the nodes that should be removed
		Graph g;
		vector<vector<int>> network = read_graph(g, FILE_NET, file_list, seed, nseed);

		Weights.resize(int(N)-1); // the removal cost of each node
		for (int i = 0; i < int(N) - 1; i++) // for the case the cost is defined as the weight !!! the id of node starts from 1, while starts from 0 in g
			Weights[i] = degree(i + 1, g); // the latter one is i+1

		vector<int> seed_final; // store the nodes that should be removed after reinsertion
		run_greedy(g, seed_final, seed, nseed); // reinsertion -- unit/unweighted case
		
		seed_final = sort_nodes_Weights(Weights, seed_final); // sort the nodes in the set nodes
		double cost = remove_write(network, seed_final, file_names[nameSub]); // cost is ratio.

		nodes_size.push_back(int(seed_final.size()));
		nodes_cost.push_back(cost);
		seed_set.push_back(seed_final);
		cout << file_list << "  " << cost <<"  "<< seed_final.size() << "  " << "\n";
	}

	vector<int>::iterator min_size = min_element(nodes_size.begin(), nodes_size.end());
	//cout << "min element at: " << std::distance(nodes_size.begin(), result);

	// vector<int>::iterator min_size = min_element(ListSize.begin(), ListSize.end());
	vector<double>::iterator min_cost = min_element(nodes_cost.begin(), nodes_cost.end());

	cout << "minimal set size is: " << *min_size << "\n";
	cout << "minimal removal cost is: " << *min_cost;
	// cout << "minimal set size is: " << *min_size << ", at: " << distance(begin(nodes_size), min_size) << "\n";
	// cout << "minimal removal cost is: " << *min_cost << ", at: " << distance(begin(nodes_cost), min_cost);

	return 0;
}
