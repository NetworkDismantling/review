#ifndef GND_H
#define GND_H

#include <vector>
#include <boost/program_options.hpp>

using namespace boost;
using namespace std;

namespace po = boost::program_options;
namespace params {
    extern unsigned int NODE_NUM;           // the number of nodes
    extern string FILE_NET;                 // input format of each line: id1 id2
    extern string FILE_ID;                  // output the id of the removed nodes in order
    extern unsigned int TARGET_SIZE;        // If the gcc size is smaller than TARGET_SIZE, the dismantling will stop. Default value can be 0.01*NODE_NUM  OR  1000
    extern int REMOVE_STRATEGY;             // 1: weighted method: powerIterationB(); vertex_cover_2() -- remove the node with smaller degree first
    extern int SEED;                    // random seed (__s default value was 1. Made it explicit)
}

using Matrix = vector<vector<unsigned int> *>*;

void rdata(const Matrix A);
void multiplyByLaplacian(const vector<vector<unsigned int> *> *A, const vector<double> *x, vector<double> *y, unsigned int dmax);
void multiplyByWeightLaplacian(const Matrix A, const vector<double> *x, vector<double> *y, const vector<unsigned int> *db, unsigned int dmax);
void orthonormalize(vector<double> *x);
vector<unsigned int> get_gcc(const Matrix adj);
vector<double> power_iteration(Matrix adj);
vector<double> power_iterationB(Matrix adj);
vector<unsigned int> vertex_cover(const Matrix A_cover, const vector<unsigned int> &degree);
vector<unsigned int> vertex_cover_2(Matrix A_cover, Matrix A_new_gcc);
void remove_nodes(Matrix A_new, vector<unsigned int> flag, vector<unsigned int> *nodes_id);
void write(const vector<unsigned int> *nodes_id);
void release_memory(Matrix *adj);
po::variables_map parse_command_line(int ac, char **av);
Matrix getMatrix(const unsigned int n);

#endif // GND_H