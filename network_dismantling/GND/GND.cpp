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

#include <iostream>
#include <fstream>
#include <set>
#include <vector>
#include <cmath>
#include <random>
#include <cstdlib>
#include <ctime>

#include <boost/program_options.hpp>

#include "GND.h"

using namespace boost;
using namespace std;

namespace po = boost::program_options;

namespace params {
    unsigned int NODE_NUM;           // the number of nodes
    string FILE_NET;                 // input format of each line: id1 id2
    string FILE_ID;                  // output the id of the removed nodes in order
    unsigned int TARGET_SIZE;        // If the gcc size is smaller than TARGET_SIZE, the dismantling will stop. Default value can be 0.01*NODE_NUM  OR  1000
    int REMOVE_STRATEGY;             // 1: weighted method: powerIterationB(); vertex_cover_2() -- remove the node with smaller degree first
    int SEED = 1;                    // random seed (__s default value was 1. Made it explicit)
}

using namespace params;


// read the links of the network, return A
void rdata(const Matrix A) {
    ifstream rd(FILE_NET.c_str());
    if (!rd) std::cerr << "Error opening network file\n";

    unsigned int id1 = 0, id2 = 0;
    while (rd >> id1 >> id2) {
        A->at(id1 - 1)->push_back(id2);
        A->at(id2 - 1)->push_back(id1);
    }

    rd.close();
}

void multiplyByLaplacian(const Matrix A, const vector<double> *x, vector<double> *y, unsigned int dmax) {
    // y = L^tilda * x
    // y_i = sum_j L^tilda_{i,j} * x_j
    // y_i = sum_j (d_max - (d_i - A_ij)) * x_j
    for (unsigned int i = 0; i < A->size(); ++i) {
        y->at(i) = 0;
        // y_i = sum_j A_ij * x_j
        for (unsigned int j = 0; j < A->at(i)->size(); ++j) {
            y->at(i) = y->at(i) + x->at(A->at(i)->at(j) - 1);
        }
        // y_i = (dmax - d_i)*x_j  + y_i
        // y_i = x_i * (dmax - degree_i) + y_i
        y->at(i) = x->at(i) * static_cast<double>(dmax - A->at(i)->size()) + y->at(i);
    }
}

void multiplyByWeightLaplacian(const Matrix A,
    const vector<double> *x, vector<double> *y,
    const vector<unsigned int> *db, unsigned int dmax
    ) {
    // y = L^tilda * x
    // y_i = sum_j (c-L_ij) * x_j

    // y_i = sum_j { A_ij*(di-1)*x_j }
    for (unsigned int i = 0; i < A->size(); ++i) {
        y->at(i) = 0;
        // y_i = A_ij * x_j
        for (unsigned int j = 0; j < A->at(i)->size(); ++j) {
            y->at(i) = y->at(i) + x->at(A->at(i)->at(j) - 1);  // y_i = sum x_j
        }
        // y_i = (d_i - 1) * y_i
        y->at(i) = static_cast<double>(A->at(i)->size() - 1) * y->at(i);
    }

    //
    for (unsigned int i = 0; i < A->size(); ++i) {
        for (unsigned int j = 0; j < A->at(i)->size(); ++j) {
            y->at(i) = y->at(i) +
                       x->at(A->at(i)->at(j) - 1) * static_cast<double>(A->at(A->at(i)->at(j) - 1)->size());
        }
        y->at(i) = y->at(i) + (dmax - db->at(i)) * x->at(i);
    }
}

void orthonormalize(vector<double> *x) {
    double inner = 0;
    unsigned int n = x->size();
    for (unsigned int no = 0; no < n; ++no) {
        inner = inner + x->at(no) / sqrt(n);
    }

    double norm = 0;
    for (unsigned int no = 0; no < n; ++no) {
        x->at(no) = x->at(no) - inner / sqrt(n);
        norm = norm + x->at(no) * x->at(no);
    }
    norm = sqrt(norm);
    for (unsigned int no = 0; no < n; ++no) {
        x->at(no) = x->at(no) / norm;
    }
}

// return a vector [transfer] that mark all the nodes belongs to gcc
// if transfer[i] = 0 then this node doesn't belong to the gcc
// if transfer[i] != 0 then transfer[i] is the new id of this node
vector<unsigned int> get_gcc(const Matrix adj) {
    const unsigned int n = adj->size();
    vector<unsigned int> id(n, 0); // store the cluster id of each node

    unsigned id_now = 0;
    for (unsigned int i = 0; i < n; i++) // wide-first search, assign each connected cluster an id
    {
        if (id[i] == 0 &&
            !adj->at(i)->empty()
            ) { // this node does not belong to any cluster yet && this node is not isolated
            set<unsigned int> set_nodes;
            set_nodes.insert(i + 1);
            id_now++;
            while (!set_nodes.empty()) {
                unsigned int node_now = *(--set_nodes.end());
                id[node_now - 1] = id_now;

                set_nodes.erase(--set_nodes.end());  // erase
                for (unsigned int k = 0; k < adj->at(node_now - 1)->size(); k++) // append
                    if (id[adj->at(node_now - 1)->at(k) - 1] == 0 &&
                        !adj->at(adj->at(node_now - 1)->at(k) - 1)->empty())
                        set_nodes.insert(adj->at(node_now - 1)->at(k));
            }
        }
    }

    unsigned int max_id = 0; // store the max id of the connected clusters
    for (unsigned int i = 0; i < n; i++)
        if (max_id < id[i]) max_id = id[i];

    vector<unsigned int> transfer(n, 0);
    if (max_id != 0) {  // max_id == 0 means the network is not connected, i.e. all the nodes are isolated
        vector<unsigned int> count(max_id, 0);
        for (unsigned int i = 0; i < n; i++)
            if (id[i] != 0)
                count[id[i] - 1]++;

        unsigned int max_size = 0; // store the size of the cluster with most nodes
        unsigned int max_cluser_id = 0; // store the id of the cluster with most nodes
        for (unsigned int i = 0; i < max_id; i++) // find the cluster with most nodes
            if (max_size < count[i]) {
                max_size = count[i];
                max_cluser_id = i + 1;
            }

        id_now = 0;
        for (unsigned int i = 0; i < n; i++) {
            if (id[i] == max_cluser_id)
                transfer[i] = ++id_now;
        }
    }
    return transfer;
}

// return eigenvector
vector<double> power_iteration(Matrix adj) {
    std::default_random_engine generator(SEED);
    std::uniform_real_distribution distribution(-1.0, 1.0);

    srand(time(nullptr));

    unsigned int n = adj->size();

    vector<double> x(n);
    vector<double> y(n);

    for (unsigned int i = 0; i < n; ++i) {
        x.at(i) = distribution(generator);
        y.at(i) = distribution(generator);
    }

    unsigned int dmax = 0;
    for (unsigned int i = 0; i < n; ++i) {
        if (adj->at(i)->size() > dmax) {
            dmax = adj->at(i)->size();
        }
    }
//    cout << "dmax: " << dmax << endl;
//    cout << "30 * log(n) * sqrt(log(n)): " << 30 * log(n) * sqrt(log(n)) << endl;
    for (unsigned int i = 0; i < 30 * log(n) * sqrt(log(n)); ++i) {

        multiplyByLaplacian(adj, &x, &y, dmax);
        multiplyByLaplacian(adj, &y, &x, dmax);
        orthonormalize(&x);
        //if (i % 30 == 0) cout << i << " -- " << 30 * log(n) * sqrt(log(n)) << endl;
    }

    return x;
}

// return eigenvector B = WA+AW-A
vector<double> power_iterationB(Matrix adj) {
    std::default_random_engine generator(SEED);
    std::uniform_real_distribution distribution(-1.0, 1.0);

    vector<double> x(adj->size());
    vector<double> y(adj->size());
    vector<unsigned int> db(adj->size());
    unsigned int n = adj->size();

    srand(time(nullptr));

    for (unsigned int i = 0; i < n; ++i) {
        x.at(i) = distribution(generator);
        y.at(i) = distribution(generator);
    }

    unsigned int dmax = 0,
                 dmax2 = 0;
    for (unsigned int i = 0; i < n; ++i) {
        auto node_i = adj->at(i);
        auto size = node_i->size();
        db.at(i) = node_i->size() * (size - 1);

        for (unsigned int j = 0; j < size; ++j) {
            db.at(i) = db.at(i) + adj->at(node_i->at(j) - 1)->size();
        }
        if (size > dmax) {
            dmax = node_i->size();
        }
        if (db.at(i) > dmax2) {
            dmax2 = db.at(i);
        }
    }
    dmax = dmax * dmax + dmax2;
    for (int i = 0; i < 30 * log(n) * sqrt(log(n)); ++i) { // 30*log(n)*log(n)
        multiplyByWeightLaplacian(adj, &x, &y, &db, dmax);
        multiplyByWeightLaplacian(adj, &y, &x, &db, dmax);
        orthonormalize(&x);
    }

    return x;
}

// return the removing order of the nodes: 1,2,3,... The node with flag=0 will not be removed
// Clarkson's Greedy Algorithm for weighted set cover
vector<unsigned int> vertex_cover(const Matrix A_cover, const vector<unsigned int> &degree) {
    vector<unsigned int> flag(A_cover->size(), 0);
    unsigned int remove = 0;
    unsigned int total_edge = 0;

    for (auto &i: *A_cover)
        total_edge += i->size();

    cout << "total_edge: " << total_edge << " before vertex cover" << endl;

    while (total_edge > 0) {
        vector<unsigned int> degree_cover(A_cover->size(), 0);
        for (unsigned int i = 0; i < A_cover->size(); i++)
            degree_cover[i] = A_cover->at(i)->size();

        vector<double> value(A_cover->size(), 0);
        for (unsigned int i = 0; i < A_cover->size(); i++)
            if (degree_cover[i] == 0)
                value[i] = 999999;
            else
                value[i] = static_cast<double>(degree[i]) / static_cast<double>(degree_cover[i]);

        double min_v = 999999;
        unsigned int min_sub = 0;
        for (unsigned int i = 0; i < value.size(); i++)
            if (min_v > value[i]) {
                min_v = value[i];
                min_sub = i;
            }
        flag[min_sub] = ++remove;
        A_cover->at(min_sub)->clear();
        for (auto &i: *A_cover)
            for (auto it = i->begin(); it != i->end();) {
                if (*it == min_sub + 1) {
                    it = i->erase(it);
                } else {
                    ++it;
                }
            }
        degree_cover[min_sub] = 0;
        total_edge = 0;
        for (auto &i: *A_cover)
            total_edge += i->size();
    }
    cout << "total_edge: " << total_edge << " after vertex cover" << endl;
    return flag;
}

// Comparing with vertex_cover, this function use the adaptive degree from the original network
// remove the node with min(degree/degree_cover) first
// return the removing order of the nodes: 1,2,3,... The node with flag=0 will not be removed
vector<unsigned int> vertex_cover_2(Matrix A_cover, Matrix A_new_gcc) {
    auto *A_new_gcc_copy = new vector<vector<unsigned int> *>(A_new_gcc->size());

    for (unsigned int i = 0; i < A_new_gcc->size(); i++) {
        auto node_i = A_new_gcc->at(i);
        auto node_i_size = node_i->size();

        A_new_gcc_copy->at(i) = new vector<unsigned int>(node_i_size);

        for (unsigned int j = 0; j < node_i_size; j++) {
            A_new_gcc_copy->at(i)->at(j) = node_i->at(j);
        }
    }

    vector<unsigned int> flag(A_cover->size(), 0); // store the cover (removal) order of each node: 1,2,3...
    unsigned int remove = 0;
    unsigned int total_edge = 0;  // the total number of edges in A_cover
    for (auto &i: *A_cover)
        total_edge += i->size();

    while (total_edge > 0) {
        vector<unsigned int> degree(A_new_gcc_copy->size(), 0);
        for (unsigned int i = 0; i < A_new_gcc_copy->size(); i++) {
            degree[i] = A_new_gcc_copy->at(i)->size();
        }
        vector<unsigned int> degree_cover(A_cover->size(), 0);
        for (unsigned int i = 0; i < A_cover->size(); i++)
            degree_cover[i] = A_cover->at(i)->size();

        vector<double> value(A_cover->size(), 0);
        for (unsigned int i = 0; i < A_cover->size(); i++)
            if (degree_cover[i] == 0)
                value[i] = 99999;
            else
                value[i] = static_cast<double>(degree[i]) / static_cast<double>(degree_cover[i]);

        double min_v = 9999;
        unsigned int min_sub = 0;
        for (unsigned int i = 0; i < value.size(); i++)
            if (min_v > value[i]) {
                min_v = value[i];
                min_sub = i;
            }
        flag[min_sub] = ++remove;
        A_cover->at(min_sub)->clear();
        A_new_gcc_copy->at(min_sub)->clear();
        for (auto &i: *A_cover)
            for (auto it = i->begin(); it != i->end();) {
                if (*it == min_sub + 1) {
                    i->erase(it);
                    it = i->begin();
                } else ++it;
            }

        for (auto &i: *A_new_gcc_copy)
            for (auto it = i->begin(); it != i->end();) {
                if (*it == min_sub + 1) {
                    i->erase(it);
                    it = i->begin();
                } else ++it;
            }

        // degree_cover[min_sub] = 0;
        total_edge = 0;
        for (auto &i: *A_cover)
            total_edge += i->size();
    }

    release_memory(&A_new_gcc_copy);

    return flag;
}

// Remove nodes from the network A_new according to flag. The removed nodes will be store in nodes_id
void remove_nodes(Matrix A_new, vector<unsigned int> flag, vector<unsigned int> *nodes_id) {
    bool flag_size = false; // continue to remove?
    unsigned int target = 0;
    for (unsigned int k = 0; k < flag.size(); k++) {
        if (flag[k] != 0) {   // set target as the first removed node
            flag_size = true; // continue to remove
            target = k;
            break;
        }
    }

    while (flag_size) { // continue to remove?
        flag_size = false;
        if (REMOVE_STRATEGY == 1) { // weighted case: find the node with minimum degree
            for (unsigned int k = 0; k < flag.size(); k++) {
                if (flag[k] != 0 && A_new->at(k)->size() < A_new->at(target)->size()) // compare the degree
                    target = k;
            }
        } else if (REMOVE_STRATEGY == 3) { // unweighted case: find the node with maximum degree
            for (unsigned int k = 0; k < flag.size(); k++) {
                if (flag[k] != 0 && A_new->at(k)->size() > A_new->at(target)->size()) // compare the degree
                    target = k;
            }
        }

        unsigned int i = target;
        auto transfer = get_gcc(A_new);
        if (flag[i] > 0 && transfer[i] != 0) {  // remove one node if the node in the remove list && the node in the gcc
            nodes_id->push_back(i + 1);
            A_new->at(i)->clear();

            for (auto &j: *A_new) {
                for (auto it = j->begin(); it != j->end();) {
                    if (*it == i + 1) {
                        j->erase(it);
                        it = j->begin();
                    } else ++it;
                }
            }
        }
        flag[target] = 0;

        for (unsigned int k = 0; k < flag.size(); k++) {
            if (flag[k] != 0) {  // set the target as the first removed node
                flag_size = true; // continue to remove
                target = k;
                break;
            }
        }

        if (!flag_size) { // reach the end of this round
            vector<unsigned int> transfer = get_gcc(A_new); // transfer has the same with A_new
            unsigned int gcc_size = 0;
            for (unsigned int k = 0; k < A_new->size(); k++)
                if (transfer[k] != 0)
                    gcc_size++;

            std::cerr << "gcc size after this round's partition - " << gcc_size << "\n";
        }
    }
}

// Output the list of nodes that should be removed in order
void write(const vector<unsigned int> *nodes_id) {
    ofstream wt_file(FILE_ID.c_str());
    if (!wt_file) std::cout << "error creating file...\n";

    for (auto i: *nodes_id)
//        std::cout << nodes_id->at(i) << endl;
        wt_file << i << endl;
    // wt_file << "S " << nodes_id->at(i) << endl;
    wt_file.close();
}

void release_memory(Matrix *adj) {
    if (*adj == nullptr)
        return;

    for (auto &i: **adj)
        delete i;

    delete (*adj);

    *adj = nullptr;
}

po::variables_map parse_command_line(int ac, char **av) {
    po::options_description desc("Usage: " + string(av[0]) + " <option> ... \n\twhere <option> is one or more of");

    desc.add_options()
            ("help", "produce help message")
            ("NetworkFile,NF", po::value(&FILE_NET), "File containing the network")
            ("IDFile,IF", po::value(&FILE_ID), "File containing the network")
//            ("OutFile,OF", po::value(&FILE_OUTPUT), "File containing the network")
            ("NodeNum,N", po::value(&NODE_NUM), "Number of vertices of the network")
            ("TargetSize,t", po::value(&TARGET_SIZE), "Target component size")
//            ("seed,s", po::value(&rdseed)->default_value(rdseed), "Pseudo-random number generator seed")
            ("RemoveStrategy,RS", po::value(&REMOVE_STRATEGY), "Removal strategy");

    po::variables_map vm;
    po::store(po::parse_command_line(ac, av, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        exit(1);
    }

    return vm;
}

Matrix getMatrix(const unsigned int n) {
    auto *A = new vector<vector<unsigned int> *>(n);

    for (unsigned int i = 0; i < n; i++)
        A->at(i) = new vector<unsigned int>();

    return A;
}


int main(int argc, char **argv) {
    po::variables_map vm = parse_command_line(argc, argv);

    //**** read adjacency matrix from file  ****
    auto *A = getMatrix(NODE_NUM);

    rdata(A);

    // // the elements' number of transfer_initial equals the number of nodes in A
    // vector<unsigned int> transfer_initial = get_gcc(A);

//     unsigned int node_size = 0;
// //    double link_size = 0;
//     for (auto i: transfer_initial)
//         if (i != 0)
//             node_size++;

    // define A_new as the gcc of A
    auto *A_new = A;
//  auto *A_new = new vector<vector<int> *>(NODE_NUM);
//	for (int i = 0; i < NODE_NUM; i++)
//		A_new->at(i) = new vector<int>();
//	for (int i = 0; i < int(transfer_initial.size()); i++)
//		for (int j = 0; j < int(A->at(i)->size()); j++) {
//			if (transfer_initial[A->at(i)->at(j) - 1] != 0) {
//				A_new->at(transfer_initial[i] - 1)->push_back(transfer_initial[A->at(i)->at(j) - 1]);
//				link_size++;
//			}
//		}
//	link_size = link_size / 2;
// 	std::cerr << "total nodes: " << node_size << " total links: " << link_size << endl;

    //**** partition the network to subnets ****
    auto *nodes_id = new vector<unsigned int>(); // store the nodes that should be removed

    // the elements' number of transfer equals the number of nodes in A
    // if transfer[i] = 0 then this node doesn't belong to the gcc
    // if transfer[i] != 0 then transfer[i] is the new id of this node in A_new_gcc
    auto transfer = get_gcc(A_new);

    unsigned int gcc_size = 0;
    for (unsigned int i = 0; i < A_new->size(); i++) {
        if (transfer[i] != 0) {
            gcc_size++;
        }
    }

    while (gcc_size > TARGET_SIZE) {
        vector<unsigned int> transfer_back(gcc_size, 0);
        for (unsigned int i = 0; i < gcc_size; i++)
            for (unsigned int j = 0; j < A_new->size(); j++) {
                if (transfer[j] == i + 1) {
                    transfer_back[i] = j + 1;
                    break;
                }
            }

        // define A_new_gcc as the gcc of A_new
        auto A_new_gcc = getMatrix(gcc_size);
        for (unsigned int i = 0; i < transfer.size(); i++) {
            if (transfer[i] > 0) {
                auto node_i = A_new->at(i);
                for (unsigned int j : *node_i) {
                    if (transfer[j - 1] > 0)
                        A_new_gcc->at(transfer[i] - 1)->push_back(transfer[j - 1]);
                }
            }
        }

        // compute the eigenvector and separate set
        vector<double> eigenvector;
        if (REMOVE_STRATEGY == 1)
            eigenvector = power_iterationB(A_new_gcc);  // L = D_B -B where B = AW + WA - A
        else if (REMOVE_STRATEGY == 3)
            eigenvector = power_iteration(A_new_gcc);   // L = D_B -B where B = A

        vector<unsigned int> flag; // mark all the nodes that should be removed to partition the network into subnet
        // flag: 0: do not remove; 1,2,3... removal order
        if (REMOVE_STRATEGY == 1 || REMOVE_STRATEGY == 3) {
            // Weighted Vertex Cover
//            auto *A_new_gcc_cover = new vector<vector<int> *>(int(A_new_gcc->size()));
//            for (int i = 0; i < gcc_size; i++) {
//                A_new_gcc_cover->at(i) = new vector<int>(); // the subnet that all the links in it should be covered
//            }

            // the subnet that all the links in it should be covered
            auto *A_new_gcc_cover = getMatrix(A_new_gcc->size());

            for (unsigned int i = 0; i < A_new_gcc->size(); i++) {
                auto node_i = A_new_gcc->at(i);
                for (unsigned int node_j : *node_i) {
                    if ((i + 1) < node_j &&  // Prevention of repeated calculation
                        eigenvector[i] * eigenvector[node_j - 1] < 0) {
                        // these two nodes do not in the same cluster
                        A_new_gcc_cover->at(i)->push_back(node_j);
                        A_new_gcc_cover->at(node_j - 1)->push_back(i + 1);
                        }
                }
            }

            cout << "A_new_gcc_cover size: " << A_new_gcc_cover->size() << endl;
            cout << "gcc_size: " << gcc_size << endl;
            if (REMOVE_STRATEGY == 1) {
                flag = vertex_cover_2(A_new_gcc_cover,
                                      A_new_gcc); // flag marks all the nodes that should be removed to partition the network into subnet
            }
            else if (REMOVE_STRATEGY == 3) {
                vector<unsigned int> degree_one(A_new_gcc->size(), 1);
                flag = vertex_cover(A_new_gcc_cover,
                                    degree_one); // flag marks all the nodes that should be removed to partition the network into subnet
            }

            release_memory(&A_new_gcc_cover);
            release_memory(&A_new_gcc);
        }

        // remove nodes
        vector<unsigned int> flag_original(A_new->size(), 0);
        for (unsigned int i = 0; i < flag.size(); i++)
            if (flag[i] != 0)
                flag_original[transfer_back[i] - 1] = flag[i];

        remove_nodes(A_new, flag_original, nodes_id);

        transfer = get_gcc(A_new);

        gcc_size = 0;
        for (unsigned int i = 0; i < A_new->size(); i++)
            if (transfer[i] != 0)
                gcc_size++;
    }

    // output the nodes that should be removed
    write(nodes_id);

    // release_memory(&A);
    // release_memory(&A_new);
    //
    // delete nodes_id;

    return 0;
}
