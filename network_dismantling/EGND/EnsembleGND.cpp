// EnsembleGND.cpp : This file contains the 'main' function. Program execution begins and ends there.

/*
 * Copyright (C) 2019 Xiao-Long Ren, Niels Gleinig, Dirk Helbing, Nino Antulov-Fantulin
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
 * This code repeatedly partitions the gcc (giant connected component) of
 * the network into two subnets by the spectcal clustering and Weighted
 * Vertex Cover algorithms, such that the size of the gcc is smaller than
 * a specific value. The output is the set of nodes that should be removed.
 * */

// TODO: After getting the random numbers, use Parallel programming to accelerate the running speed

#include <iostream>
#include <fstream>
#include <sstream>
#include <set>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <random>
#include <cstdlib>
#include <ctime>

#include <chrono>

// START Edit by M. Grassia to fix compilation
#include <cfloat>
#include <algorithm>
// END

#include "config.h"


using namespace std;


// read the links of the network, return A
void rdata(vector<vector<int>*>* A) {
	ifstream rd(FILE_NET);
	if (!rd) std::cout << "error opening file\n";

	int id1 = 0, id2 = 0;
	while (rd >> id1 >> id2) {
		A->at(id1 - 1)->push_back(id2);
		A->at(id2 - 1)->push_back(id1);
	}
	rd.close();
}

void release_memory(vector<vector<int>*>* adj) {
	for (int i = 0; i < adj->size(); ++i) {
		delete adj->at(i);
	}
}

void multiplyByLaplacian(vector<vector<int>*>* A, vector<double>* x, vector<double>* y, int dmax)
{
	// y = L^tilda * x
	// y_i = sum_j L^tilda_{i,j} * x_j
	// y_i = sum_j (d_max - (d_i - A_ij)) * x_j
	for (int i = 0; i < A->size(); ++i) {
		y->at(i) = 0;
		// y_i = sum_j A_ij * x_j
		for (int j = 0; j < int(A->at(i)->size()); ++j) {
			y->at(i) = y->at(i) + x->at(A->at(i)->at(j) - 1);
		}
		// y_i = (dmax - d_i)*x_j  + y_i 
		// y_i = x_i * (dmax - degree_i) + y_i
		y->at(i) = x->at(i) * (dmax - int(A->at(i)->size())) + y->at(i);
	}
}

void multiplyByWeightLaplacian(vector<vector<int>*>* A, vector<double>* x, vector<double>* y, vector<int>* db, int dmax)
{
	// y = L^tilda * x
	// y_i = sum_j (c-L_ij) * x_j

	// y_i = sum_j { A_ij*(di-1)*x_j }
	for (int i = 0; i < A->size(); ++i) {
		y->at(i) = 0;
		// y_i = A_ij * x_j
		for (int j = 0; j < A->at(i)->size(); ++j) {
			y->at(i) = y->at(i) + x->at(A->at(i)->at(j) - 1);  // y_i = sum x_j
		}
		// y_i = (d_i - 1) * y_i
		y->at(i) = (A->at(i)->size() - 1) * y->at(i);
	}

	// 
	for (int i = 0; i < A->size(); ++i) {
		for (int j = 0; j < A->at(i)->size(); ++j) {
			y->at(i) = y->at(i) + x->at(A->at(i)->at(j) - 1) * A->at(A->at(i)->at(j) - 1)->size();
		}
		y->at(i) = y->at(i) + (dmax - db->at(i)) * x->at(i);
	}
}

void orthonormalize(vector<double>* x)
{
	double inner = 0;
	int n = int(x->size());
	for (int no = 0; no < n; ++no) {
		inner = inner + x->at(no) / sqrt(n);
	}

	double norm = 0;
	for (int no = 0; no < n; ++no) {
		x->at(no) = x->at(no) - inner / sqrt(n);
		norm = norm + x->at(no) * x->at(no);
	}
	norm = sqrt(norm);
	for (int no = 0; no < n; ++no) {
		x->at(no) = x->at(no) / norm;
	}
}

// return a vector [transfer] that mark all the nodes belongs to gcc
// if transter[i] = 0 then this node doesn't belong to the gcc
// if transter[i] != 0 then transter[i] is the new id of this node 
vector<int> get_gcc(vector<vector<int>*>* adj) {
	int n = int(adj->size());
	vector<int> id(n, 0); // store the cluster id of each node

	int id_now = 0;
	for (int i = 0; i < n; i++) // wide-first search, assign each connected cluster an id
	{
		if (id[i] == 0 && adj->at(i)->size() > 0) { // this node does not belong to any cluster yet && this node is not isolated
			set<int> set_nodes;
			set_nodes.insert(i + 1);
			id_now++;
			while (set_nodes.size() > 0)
			{
				int node_now = *(--set_nodes.end());
				id[node_now - 1] = id_now;

				set_nodes.erase(--set_nodes.end());  // erase
				for (int k = 0; k<int(adj->at(node_now - 1)->size()); k++) // append
					if (id[adj->at(node_now - 1)->at(k) - 1] == 0 && adj->at(adj->at(node_now - 1)->at(k) - 1)->size() != 0)
						set_nodes.insert(adj->at(node_now - 1)->at(k));
			}
		}
	}

	int max_id = 0; // store the max id of the connected clusters
	for (int i = 0; i < n; i++)
		if (max_id < id[i]) max_id = id[i];

	vector<int> transfer(n, 0);
	if (max_id != 0) {  // max_id == 0 means the network is not connected, i.e. all the nodes are isolated
		vector<int> count(max_id, 0);
		for (int i = 0; i < n; i++)
			if (id[i] != 0)
				count[id[i] - 1]++;

		int max_size = 0; // store the size of the cluster with most nodes
		int max_cluser_id = 0; // store the id of the cluster with most nodes
		for (int i = 0; i < max_id; i++) // find the cluster with most nodes
			if (max_size < count[i]) {
				max_size = count[i];
				max_cluser_id = i + 1;
			}

		id_now = 0;
		for (int i = 0; i < n; i++) {
			if (id[i] == max_cluser_id)
				transfer[i] = ++id_now;
		}
	}
	return transfer;
}

// return eigenvector
vector<double> power_iteration(vector<vector<int>*>* adj, double C) {
	std::mt19937 generator(C);
	std::uniform_real_distribution<double> distribution(-1.0, 1.0);

	vector<double> x(int(adj->size()));
	vector<double> y(int(adj->size()));
	int n = int(adj->size());

	for (int i = 0; i < n; ++i) {
		x.at(i) = distribution(generator);
		y.at(i) = distribution(generator);
	}

	int dmax = 0;
	for (int i = 0; i < n; ++i) {
		if (int(adj->at(i)->size()) > dmax) {
			dmax = int(adj->at(i)->size());
		}
	}
	// orthonormalize(&x);
	for (int i = 0; i < 30 * log(n) * sqrt(log(n)); ++i) {
		multiplyByLaplacian(adj, &x, &y, 2 * dmax);
		orthonormalize(&y);
		multiplyByLaplacian(adj, &y, &x, 2 * dmax);
		orthonormalize(&x);
		
		int diff = 0;
		for (int j = 0; j<int(x.size()); j++) {
			if (y.at(j) != x.at(j)) diff++;
		}
		if (diff == 0) break;
	}

	return x;
}

// return eigenvector B = WA+AW-A
vector<double> power_iterationB(vector<vector<int>*>* adj, double C) {
	std::mt19937 generator(C);
	std::uniform_real_distribution<double> distribution(-1.0, 1.0);

	vector<double> x(adj->size());
	vector<double> y(adj->size());
	vector<int> db(adj->size());
	int n = int(adj->size());

	for (int i = 0; i < n; ++i) {
		x.at(i) = distribution(generator);
		y.at(i) = distribution(generator);
	}

	int dmax = 0;
	int dmax2 = 0;
	for (int i = 0; i < n; ++i) {
		db.at(i) = int(adj->at(i)->size()) * int((adj->at(i)->size() - 1));
		for (int j = 0; j < adj->at(i)->size(); ++j) {
			db.at(i) = db.at(i) + int(adj->at(adj->at(i)->at(j) - 1)->size());
		}
		if (adj->at(i)->size() > dmax) {
			dmax = int(adj->at(i)->size());
		}
		if (db.at(i) > dmax2) {
			dmax2 = db.at(i);
		}
	}
	dmax = dmax * dmax + dmax2;
	for (int i = 0; i < 30 * log(n) * sqrt(log(n)); ++i) {
		multiplyByWeightLaplacian(adj, &x, &y, &db, dmax);
		orthonormalize(&y);
		multiplyByWeightLaplacian(adj, &y, &x, &db, dmax);
		orthonormalize(&x);

		int diff = 0;
		for (int j = 0; j<int(x.size()); j++) {
			if (y.at(j) != x.at(j)) diff++;
		}
		if (diff == 0) 	break;
	}

	return x;
}

// adjust the group partition of nodes if all its neighbors are in a different group
void adjust_eig(vector<vector<int>*>* adj, vector<double>& eig) {
	double negative_count = 0, postive_count = 0;
	for (int i = 0; i<int(eig.size()); i++) {
		if (eig[i] < 0) negative_count++;
		else if (eig[i] >= 0) postive_count++; // here is >=
	}
	for (int i = 0; i<int(adj->size()); i++) {
		bool same_group = false;
		for (int j = 0; j<int(adj->at(i)->size()); j++) {
			if (eig[i] * eig[adj->at(i)->at(j) - 1] >= 0) // i and j are not in the same group
				same_group = true;
		}
		if (!same_group && int(adj->at(i)->size()) != 0 && negative_count > 1 && postive_count > 1) { // >1 is to prevent all the nodes are in the same group after adjust
			if (eig[i] < 0) negative_count--;
			else postive_count--;
			eig[i] = eig[adj->at(i)->at(0) - 1]; // ajust node i to another group
		}
	}
}

// return the removing order of the nodes: 1,2,3,... The node with flag=0 will not be removed
// Clarkson's Greedy Algorithm for weighted set cover
vector<int> vertex_cover(vector<vector<int>*>* A_cover, vector<int> degree) {
	vector <int> flag(int(A_cover->size()), 0);
	int remove = 0;
	int total_edge = 0;
	for (int i = 0; i < int(A_cover->size()); i++)
		total_edge += int(A_cover->at(i)->size());

	while (total_edge > 0) {
		vector<int> degree_cover(int(A_cover->size()), 0);
		for (int i = 0; i < int(A_cover->size()); i++)
			degree_cover[i] = int(A_cover->at(i)->size());

		vector<double> value(int(A_cover->size()), 0);
		for (int i = 0; i < int(A_cover->size()); i++)
			if (degree_cover[i] == 0)
				value[i] = DBL_MAX;
			else
				value[i] = double(degree[i]) / double(degree_cover[i]);

		double min_v = DBL_MAX;
		int min_sub = 0;
		for (int i = 0; i<int(value.size()); i++)
			if (min_v > value[i]) {
				min_v = value[i];
				min_sub = i;
			}
		flag[min_sub] = ++remove;
		A_cover->at(min_sub)->clear();
		for (int i = 0; i < int(A_cover->size()); i++)
			for (vector<int>::iterator it = A_cover->at(i)->begin(); it != A_cover->at(i)->end(); )
			{
				if (*it == min_sub + 1) {
					A_cover->at(i)->erase(it);
					it = A_cover->at(i)->begin();
				}
				else it++;
			}
		degree_cover[min_sub] = 0;
		total_edge = 0;
		for (int i = 0; i < int(A_cover->size()); i++)
			total_edge += int(A_cover->at(i)->size());
	}
	return flag;
}

// Comparing with vertex_cover, this function use the adaptive degree from the original network
// remove the node with min(degree/degree_cover) first
// return the removing order of the nodes: 1,2,3,... The node with flag=0 will not be removed
vector<int> vertex_cover_2(vector<vector<int>*>* A_cover, vector<vector<int>*>* A_new_gcc) {
	vector<vector<int>*>* A_new_gcc_copy = new vector<vector<int>*>(int(A_new_gcc->size()));
	for (int i = 0; i < int(A_new_gcc->size()); i++) {
		A_new_gcc_copy->at(i) = new vector<int>(int(A_new_gcc->at(i)->size()));
		for (int j = 0; j<int(A_new_gcc->at(i)->size()); j++) {
			A_new_gcc_copy->at(i)->at(j) = A_new_gcc->at(i)->at(j);
		}
	}

	vector <int> flag(int(A_cover->size()), 0); // store the cover (removal) order of each node: 1,2,3...
	int remove = 0;
	int total_edge = 0;  // the total number of edges in A_cover
	for (int i = 0; i < int(A_cover->size()); i++)
		total_edge += int(A_cover->at(i)->size());

	while (total_edge > 0) {
		vector<int> degree(int(A_new_gcc_copy->size()), 0);
		for (int i = 0; i < int(A_new_gcc_copy->size()); i++) {
			degree[i] = int(A_new_gcc_copy->at(i)->size());
		}
		vector<int> degree_cover(int(A_cover->size()), 0);
		for (int i = 0; i < int(A_cover->size()); i++)
			degree_cover[i] = int(A_cover->at(i)->size());

		vector<double> value(int(A_cover->size()), 0);
		for (int i = 0; i < int(A_cover->size()); i++)
			if (degree_cover[i] == 0)
				value[i] = DBL_MAX;
			else
				value[i] = double(degree[i]) / double(degree_cover[i]);

		double min_v = DBL_MAX;
		int min_sub = 0;
		for (int i = 0; i<int(value.size()); i++)
			if (min_v > value[i]) {
				min_v = value[i];
				min_sub = i;
			}
		flag[min_sub] = ++remove;
		A_cover->at(min_sub)->clear();
		A_new_gcc_copy->at(min_sub)->clear();
		for (int i = 0; i < int(A_cover->size()); i++)
			for (vector<int>::iterator it = A_cover->at(i)->begin(); it != A_cover->at(i)->end(); )
			{
				if (*it == min_sub + 1) {
					A_cover->at(i)->erase(it);
					it = A_cover->at(i)->begin();
				}
				else it++;
			}

		for (int i = 0; i < int(A_new_gcc_copy->size()); i++)
			for (vector<int>::iterator it = A_new_gcc_copy->at(i)->begin(); it != A_new_gcc_copy->at(i)->end(); )
			{
				if (*it == min_sub + 1) {
					A_new_gcc_copy->at(i)->erase(it);
					it = A_new_gcc_copy->at(i)->begin();
				}
				else it++;
			}

		// degree_cover[min_sub] = 0;
		total_edge = 0;
		for (int i = 0; i < int(A_cover->size()); i++)
			total_edge += int(A_cover->at(i)->size());
	}
	release_memory(A_new_gcc_copy);
	A_new_gcc_copy->clear();
	return flag;
}

// Remove nodes from the network A_new according to flag. The removed nodes will be store in nodes_id
void remove_nodes(vector<vector<int>*>* A_new, vector<int> flag, vector<double>* y_gcc, vector<double>* x_links, vector<double>* x_nodes, vector<int>* nodes_id) {
	int removed_nodes = 0, removed_links = 0;
	if (y_gcc->size() != 0) {
		removed_nodes = x_nodes->back();
		removed_links = x_links->back();
	}

	bool flag_size = false; // continue to remove?
	int target = 0;
	for (int k = 0; k<int(flag.size()); k++) {
		if (flag[k] != 0) {   // set target as the first removed node
			flag_size = true; // continue to remove
			target = k;
			break;
		}
	}

	while (flag_size) { // continue to remove?
		flag_size = false;
		if (REMOVE_STRATEGY == 1) { // weighted case: find the node with minimum degree
			for (int k = 0; k<int(flag.size()); k++) {
				if (flag[k] != 0 && A_new->at(k)->size() < A_new->at(target)->size()) // compare the degree
					target = k;
			}
		}
		else if (REMOVE_STRATEGY == 3) { // unweighted case: find the node with maximum degree
			for (int k = 0; k<int(flag.size()); k++) {
				if (flag[k] != 0 && A_new->at(k)->size() > A_new->at(target)->size()) // compare the degree
					target = k;
			}
		}

		int i = target;
		vector<int> transfer = get_gcc(A_new);
		if (flag[i] > 0 && transfer[i] != 0) {  // remove one node if the node in the remove list && the node in the gcc
			nodes_id->push_back(i + 1);
			removed_nodes++;
			removed_links += int(A_new->at(i)->size());
			A_new->at(i)->clear();

			for (int j = 0; j < int(A_new->size()); j++) {
				for (vector<int>::iterator it = A_new->at(j)->begin(); it != A_new->at(j)->end(); ) {
					if (*it == i + 1) {
						A_new->at(j)->erase(it);
						it = A_new->at(j)->begin();
					}
					else it++;
				}
			}
			if (removed_nodes % PLOT_SIZE == 0) { // record
				vector<int> transfer = get_gcc(A_new); // transfer has the same size with A_new
				int gcc_size = 0;
				for (int i = 0; i < int(A_new->size()); i++)
					if (transfer[i] != 0)
						gcc_size++;

				int temp = 0;
				for (int k = 0; k<int(A_new->size()); k++) {
					if (A_new->at(k)->size() != 0) temp++;
				}

				y_gcc->push_back(gcc_size);
				x_links->push_back(removed_links);
				x_nodes->push_back(removed_nodes);
			}
		}
		flag[target] = 0;

		for (int k = 0; k<int(flag.size()); k++) {
			if (flag[k] != 0) {  // set the target as the first removed node
				flag_size = true; // continue to remove
				target = k;
				break;
			}
		}

		if (!flag_size) { // reach the end of this round
			vector<int> transfer = get_gcc(A_new); // transfer has the same with A_new
			int gcc_size = 0;
			for (int i = 0; i < int(A_new->size()); i++)
				if (transfer[i] != 0)
					gcc_size++;
			if (PLOT_SIZE != 1) {
				y_gcc->push_back(gcc_size);
				x_links->push_back(removed_links);
				x_nodes->push_back(removed_nodes);
			}

			//std::cout << "gcc size after this round's partition - " << gcc_size << "\n";
		}
	}
}

// Output the list of nodes that should be removed in order
void write(vector<double>* y_gcc, vector<double>* x_links, vector<double>* x_nodes, vector<int>* nodes_id, double C) {
	stringstream ss, ss2;
	string FILE_ID_full, FILE_PLOT_full;
	ss << FILE_ID;
//	ss << C;
//	ss << ".txt";
	ss >> FILE_ID_full;

	ss2 << FILE_ID;
//	ss2 << C;
//	ss2 << "_plot.txt";
	ss2 >> FILE_PLOT_full;

	ofstream wt_id(FILE_ID_full), wt_plot(FILE_PLOT_full);
	if (!wt_id || !wt_plot) std::cout << "error creating file...\n";

	for (int i = 0; i<int(nodes_id->size()); i++)
		wt_id << nodes_id->at(i) << endl;
	wt_id.close();

	// cout << "\n plot file format: gcc removed_cost removed_nodes\n";
//	wt_plot << 1 << " " << 0 << " " << 0 << endl;
//	for (int i = 0; i<int(y_gcc->size()); i++)
//		wt_plot << y_gcc->at(i) << " " << x_links->at(i) << " " << x_nodes->at(i) << "\n";
//	wt_plot.close();
}

int main() {
	vector<vector<int>*>* A_orig = new vector<vector<int>*>(NODE_NUM);
	for (int i = 0; i<int(A_orig->size()); ++i)
		A_orig->at(i) = new vector<int>();
	rdata(A_orig);

	vector<double> CSeed(C, 0); // store the seeds of different run of GND
	std::mt19937 generator;
	std::uniform_real_distribution<double> distribution(1, C * C); // range of random numbers

	for (int i = 0; i < C; i++) {
		CSeed[i] = distribution(generator);
	}
	ofstream wt(FILE_SEED);
	for (int i = 0; i < C; i++) {
		wt << CSeed[i] << "\n";
	}
	wt.close();

	vector<int> ListSize(C, 0);
	vector<double> ListCost(C, 0);
	for (int C_i = 0; C_i < C; C_i++) {
		//using namespace std::chrono;
		//auto start = high_resolution_clock::now();

		vector<vector<int>*>* A = new vector<vector<int>*>(NODE_NUM); // A is a copy of A_orig
		for (int i = 0; i<int(A->size()); ++i) {
			A->at(i) = new vector<int>(A_orig->at(i)->size());
			for (int j = 0; j < int(A_orig->at(i)->size()); j++) {
				A->at(i)->at(j) = A_orig->at(i)->at(j);
			}
		}

		vector<int> transfer_initial = get_gcc(A); // the elements' number of transfer_initial equals the number of nodes in A
		double node_size = 0, link_size = 0;
		for (int i = 0; i<int(transfer_initial.size()); i++)
			if (transfer_initial[i] != 0)
				node_size++;

		// define A_new as the gcc of A
		vector<vector<int>*>* A_new = new vector<vector<int>*>(node_size);
        A_new = A;

//		for (int i = 0; i < node_size; i++)
//			A_new->at(i) = new vector<int>();
//		for (int i = 0; i < int(transfer_initial.size()); i++)
//			for (int j = 0; j < int(A->at(i)->size()); j++) {
//				if (transfer_initial[A->at(i)->at(j) - 1] != 0) {
//					A_new->at(transfer_initial[i] - 1)->push_back(transfer_initial[A->at(i)->at(j) - 1]);
//					link_size++;
//				}
//			}
//		link_size = link_size / 2;
		// std::cout << "total nodes: " << node_size << " total links: " << link_size << endl;


		//**** partation the network to subnets ****
		vector<double>* y_gcc = new vector<double>();
		vector<double>* x_links = new vector<double>();
		vector<double>* x_nodes = new vector<double>();
		vector<int>* nodes_id = new vector<int>(); // store the nodes that should be removed
		int gcc_size = int(A->size());
		while (gcc_size > TARGET_SIZE)
		{
			vector<int> transfer = get_gcc(A_new); // the elements' number of transfer equals the number of nodes in A
												   // if transter[i] = 0 then this node doesn't belong to the gcc
												   // if transter[i] != 0 then transter[i] is the new id of this node in A_new_gcc
			gcc_size = 0;
			for (int i = 0; i < int(A_new->size()); i++)
				if (transfer[i] != 0)
					gcc_size++;
			vector<int> transfer_back(gcc_size, 0);
			for (int i = 0; i < gcc_size; i++)
				for (int j = 0; j< int(A_new->size()); j++) {
					if (transfer[j] == i + 1) {
						transfer_back[i] = j + 1;
						break;
					}
				}

			// define A_new_gcc as the gcc of A_new
			vector<vector<int>*>* A_new_gcc = new vector<vector<int>*>(gcc_size);
			for (int i = 0; i < gcc_size; i++)
				A_new_gcc->at(i) = new vector<int>();
			for (int i = 0; i < int(transfer.size()); i++) {
				if (transfer[i] != 0) {
					for (int j = 0; j < int(A_new->at(i)->size()); j++) {
						if (transfer[A_new->at(i)->at(j) - 1] != 0)
							A_new_gcc->at(transfer[i] - 1)->push_back(transfer[A_new->at(i)->at(j) - 1]);
					}
				}
			}

			// compute the eigenvector and seperate set
			vector<double> eigenvector;
			if (REMOVE_STRATEGY == 1)
				eigenvector = power_iterationB(A_new_gcc, CSeed[C_i]);  // L = D_B -B where B = AW + WA - A
			else if (REMOVE_STRATEGY == 3)
				eigenvector = power_iteration(A_new_gcc, CSeed[C_i]);   // L = D_B -B where B = A
			adjust_eig(A_new_gcc, eigenvector);

			vector<int> flag; // mark all the nodes that should be removed to partition the network into subnet
							  // flag: 0: do not remove; 1,2,3.. removal order
			if (REMOVE_STRATEGY == 1 || REMOVE_STRATEGY == 3) {  // Weighted Vertex Cover
				vector<vector<int>*>* A_new_gcc_cover = new vector<vector<int>*>(int(A_new_gcc->size()));

				for (int i = 0; i < gcc_size; i++) {
					A_new_gcc_cover->at(i) = new vector<int>(); // the subnet that all the links in it should be covered
				}
				for (int i = 0; i<int(A_new_gcc->size()); i++)
					for (int j = 0; j < int(A_new_gcc->at(i)->size()); j++) {
						if ((i + 1) < A_new_gcc->at(i)->at(j) &&  // Prevention of repeated calculation
							eigenvector[i] * eigenvector[A_new_gcc->at(i)->at(j) - 1] <= 0) {// these two nodes do not in the same cluster
							A_new_gcc_cover->at(i)->push_back(A_new_gcc->at(i)->at(j));
							A_new_gcc_cover->at(A_new_gcc->at(i)->at(j) - 1)->push_back(i + 1);
						}
					}
				if (REMOVE_STRATEGY == 1) {
					flag = vertex_cover_2(A_new_gcc_cover, A_new_gcc); // flag marks all the nodes that should be removed to partition the network into subnet
				}
				else if (REMOVE_STRATEGY == 3) {
					vector<int> degree_one(int(A_new_gcc->size()), 1);
					flag = vertex_cover(A_new_gcc_cover, degree_one); // flag marks all the nodes that should be removed to partition the network into subnet
				}
			}

			// remove nodes
			vector<int> flag_orginal(int(A_new->size()), 0);
			for (int i = 0; i<int(flag.size()); i++)
				if (flag[i] != 0)
					flag_orginal[transfer_back[i] - 1] = flag[i];

			remove_nodes(A_new, flag_orginal, y_gcc, x_links, x_nodes, nodes_id);

			transfer = get_gcc(A_new);
			gcc_size = 0;
			for (int i = 0; i < int(A_new->size()); i++)
				if (transfer[i] != 0)
					gcc_size++;

			release_memory(A_new_gcc);
			A_new_gcc->clear();
		}

		for (int i = 0; i<int(y_gcc->size()); i++) {
			y_gcc->at(i) = y_gcc->at(i) / node_size;
			x_nodes->at(i) = x_nodes->at(i) / node_size;
			x_links->at(i) = x_links->at(i) / link_size;
		}

		write(y_gcc, x_links, x_nodes, nodes_id, C_i); // output the list of nodes that should be removed
		ListSize[C_i] = int(nodes_id->size());
		ListCost[C_i] = x_links->back();
//		cout << C_i << "   " << CSeed[C_i] << "   " << ListSize[C_i] << "   " << ListCost[C_i] << "\n";

		release_memory(A);
		A->clear();
		release_memory(A_new);
		A_new->clear();

		// using namespace std::chrono;
		// auto stop = high_resolution_clock::now();
		//auto duration = duration_cast<microseconds>(stop - start);
		//cout << C_i<<" "<< double(duration.count())/double(1000000) << endl;
	}

	vector<int>::iterator min_size = min_element(ListSize.begin(), ListSize.end());
	vector<double>::iterator min_cost = min_element(ListCost.begin(), ListCost.end());
	cout << "minimal set size is: " << *min_size << ", at: " << distance(begin(ListSize), min_size)<<"\n";
	cout << "minimal removal cost is: " << *min_cost << ", at: " << distance(begin(ListCost), min_cost);

//	system("pause");
	return 0;
}
