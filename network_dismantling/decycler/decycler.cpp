
/*
 * Decycler, a reinforced Max-Sum algorithm to solve the decycling problem
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


#include <boost/config.hpp>

#include "mes.hpp"

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
#include "omp_config.h"

using namespace boost;
using namespace std;

namespace params {

    int depth = 20;
    int maxit = 10000;
    int maxdec = 30;
    real_t tolerance = 1e-5;
    real_t beta_param = 1e-3;
    real_t rein = 0;
    real_t noise = 1e-7;
    real_t mu = 0.1;
    int plotting = false;
    real_t rho = 1e-5;
    int numthreads = 1;

}

using namespace params;
typedef Mes_t<2> Mes;

struct ConvexCavity {
    void reset() {
        H.clear();
        m1 = -inf;
        m2 = -inf;
        idxm1 = 0;
        SH = 0;
    }

    void push_back(real_t h0, real_t h1) {
        H.push_back(h1);
        SH += h1;
        real_t const h01 = h0 - h1;
        if (h01 >= m1) {
            m2 = m1;
            m1 = h01;
            idxm1 = H.size() - 1;
        } else if (h01 > m2) {
            m2 = h01;
        }

    }

    real_t cavityval(int i) const {
        return SH - H[i];
    }

    real_t cavitymax(int i) const {
        return SH - H[i] + max(real_t(0.), i == idxm1 ? m2 : m1);
    }

    real_t fullmax() const {
        return SH + max(real_t(0.), m1);
    }

    std::vector <real_t> H;
    real_t m1, m2, SH;
    int idxm1;
};

struct check_type {
    check_type() : num_bad(0), num_seeds(0), num_on(0), tot_cost(0), tot_energy(0) {}

    int num_bad;
    int num_seeds;
    int num_on;
    real_t tot_cost;
    real_t tot_energy;

    check_type &operator+=(check_type const b) {
        num_bad += b.num_bad;
        num_seeds += b.num_seeds;
        num_on += b.num_on;
        tot_cost += b.tot_cost;
        tot_energy += b.tot_energy;
        return *this;
    }

    friend ostream &operator<<(ostream &o, check_type const &v) {
        return o << "# Check:  num_on=" << v.num_on
                 << "  num_seeds=" << v.num_seeds
                 << "  tot_cost=" << v.tot_cost
                 << "  num_bad=" << v.num_bad
                 << "  tot_energy=" << v.tot_energy;
    }

};

struct Buffers {
    Buffers() : Ui(depth + 1), out(depth + 1), C(depth + 1), L0(depth + 1), G1(depth + 1) {}

    void init(int n) {
        Hin.resize(n, Mes(depth + 1));
        std::fill(&maxh[0], &maxh[0] + maxh.size(), -inf);
        std::fill(&maxh2[0], &maxh2[0] + maxh2.size(), -inf);
        maxh.resize(n, -inf);
        maxh2.resize(n, -inf);
        for (int t = 0; t <= depth; ++t)
            C[t].reset();
    }

    vector <Mes> Hin;
    vector <real_t> maxh;
    vector <real_t> maxh2;
    Proba Ui;
    Mes out;
    vector <ConvexCavity> C;
    vector <real_t> L0, G1;
    check_type ck;
};

vector <Buffers> Mem;


real_t mincost = inf;

mt19937 gen;
mt19937 mes_gen;

uniform_real<> const uni_dist(0, 1);
variate_generator<mt19937 &, uniform_real<> > real01(gen, uni_dist);
variate_generator<mt19937 &, uniform_real<> > mes_real01(mes_gen, uni_dist);

struct EdgeProperties {
    EdgeProperties() :
            ij(depth + 1), ji(depth + 1) {
        omp_init_lock(&lock);
    }

    //edge properties
    //messages
    Mes ij, ji;
#ifdef OMP
    omp_lock_t lock;
#endif
};

struct VertexProperties {
    VertexProperties() : H(depth + 1, 0), extH(depth + 1, 0), t(0) {}

    //VertexProperties(string const & name) :
    //name_(name),
    //H(depth + 1, 0), t(0)
    //{}
    //string const & name() const { return name_; }
    //node properties
    //string name_;
    //field
    Proba H;
    Proba extH;
    //decisional variable
    int t;
};


typedef adjacency_list <vecS, vecS, undirectedS,
VertexProperties, EdgeProperties> Graph;
typedef graph_traits<Graph>::vertex_iterator vertex_iterator;
typedef graph_traits<Graph>::out_edge_iterator edge_iterator;
typedef graph_traits<Graph>::edge_iterator graph_edge_iterator;
typedef graph_traits<Graph>::edge_descriptor Edge;
typedef graph_traits<Graph>::vertex_descriptor Vertex;

typedef adjacency_list <vecS, vecS, undirectedS> SGraph;
typedef graph_traits<SGraph>::edge_descriptor SEdge;
typedef graph_traits<SGraph>::vertex_descriptor SVertex;
typedef graph_traits<SGraph>::out_edge_iterator sedge_iterator;
Graph g;

inline void setMes(Edge e, Mes const &m) {
    omp_set_lock(&g[e].lock);
    (source(e, g) > target(e, g) ? g[e].ji : g[e].ij) = m;
    omp_unset_lock(&g[e].lock);
}

inline void getMes(Edge e, Mes &m) {
    omp_set_lock(&g[e].lock);
    m = source(e, g) < target(e, g) ? g[e].ji : g[e].ij;
    omp_unset_lock(&g[e].lock);
}


void propagate() {
    std::deque<int> q;
    vector<int> theta(num_vertices(g));
    vector<int> times(num_vertices(g));
    vertex_iterator vit, vend;
    for (tie(vit, vend) = vertices(g); vit != vend; ++vit) {
        int v = *vit;
        if (g[v].t == 0) {
            theta[v] = 0;
            times[v] = 0;
            q.push_back(v);
        } else {
            theta[v] = out_degree(v, g) - 1;
            times[v] = depth;
            if (theta[v]) {
                times[v] = 1;
                q.push_back(v);
            }
        }
    }


    while (!q.empty()) {
        int u = q.front();
        q.pop_front();
        edge_iterator eit, end;
        for (tie(eit, end) = out_edges(u, g); eit != end; ++eit) {
            int v = target(*eit, g);
            if (--theta[v] == 0) {
                times[v] = times[u] + 1;
                q.push_back(v);
            }
        }
    }

    int seeds = 0;
    int num_on = 0;
    for (unsigned i = 0; i < num_vertices(g); ++i) {
        num_on += (times[i] < depth);
        seeds += (times[i] == 0);
        g[i].t = times[i];
    }
}

void print_times() {
    cout << "# Time assignment:\n";
    for (unsigned j = 0; j < num_vertices(g); ++j)
        cout << setw(10) << j << setw(10) << g[j].t / float(depth)
             << endl;
}


void print_seeds() {
//	cout << "# Seeds" << endl;
    for (unsigned j = 0; j < num_vertices(g); ++j)
        if (g[j].t == 0)
            cout << "S " << j << endl;
}


void print_fields() {
    for (unsigned j = 0; j < num_vertices(g); ++j)
        cout << j << " " << g[j].H << endl;

}


check_type check_v(bool output = false) {
    for (unsigned p = 0; p < Mem.size(); ++p)
        Mem[p].ck = check_type();
#pragma omp parallel for
    for (unsigned j = 0; j < num_vertices(g); ++j) {
        int sp = 0;
        edge_iterator eit, eend;
        int const tj = g[j].t;
        for (tie(eit, eend) = out_edges(j, g); eit != eend; ++eit) {
            int const ti = g[target(*eit, g)].t;
            sp += (ti <= tj - 1);
        }
        check_type &ck = Mem[omp_get_thread_num()].ck;
        if (tj < depth) {
            ck.num_on++;
            ck.tot_energy--;
        }
        if (tj == 0) {
            ck.num_seeds++;
            ck.tot_cost++;
            ck.tot_energy += mu;
        }
        int th = out_degree(j, g) - 1;
        int good = (th == 0 && tj == 1)
                   || tj == 0
                   || tj == depth
                   || sp >= th;
        if (!good)
            ck.num_bad++;
    }
    check_type ck;
    for (unsigned p = 0; p < Mem.size(); ++p)
        ck += Mem[p].ck;
//	if (output)
//		cout << ck << endl;
    return ck;
}


template<class T>
ostream &operator<<(ostream &o, vector <T> const &v) {
    o << "( ";
    std::copy(v.begin(), v.end(), ostream_iterator < T const & > (o, " "));
    o << " )";
    return o;
}


real_t update(Vertex i) {
    int n = out_degree(i, g);
    if (n == 0) {
        g[i].H = Proba(depth + 1, -inf);
        g[i].H[1] = 0;
        return 0;
    }

    Buffers &buffers = Mem[omp_get_thread_num()];
    buffers.init(n);
    vector <ConvexCavity> &C = buffers.C;;
    vector <Mes> &Hin = buffers.Hin;
    vector <real_t> &maxh = buffers.maxh;
    vector <real_t> &maxh2 = buffers.maxh2;
    Mes &out = buffers.out;
    Proba &Ui = buffers.Ui;


    real_t summaxh = 0, summaxh2 = 0;

    edge_iterator eit, eend;
    int j = 0;

    for (tie(eit, eend) = out_edges(i, g); eit != eend; ++eit, ++j) {
        getMes(*eit, Hin[j]);
        Proba const *h = Hin[j].H;
        vector <real_t> &L0 = buffers.L0;
        vector <real_t> &G1 = buffers.G1;
        L0[0] = h[0][0];
        for (int ti = 1; ti <= depth; ++ti)
            L0[ti] = max(L0[ti - 1], h[0][ti]);
        G1[depth] = -inf;
        for (int ti = depth; ti--;)
            G1[ti] = max(G1[ti + 1], h[1][ti + 1]);

        for (int ti = 1; ti <= depth; ++ti) {
            real_t hww = L0[ti - 1];
            real_t hzz = max(h[0][ti], G1[ti]);
            C[ti].push_back(hzz, hww);
        }

        maxh[j] = max(h[0][0], G1[0]);
        maxh2[j] = L0[depth];
        summaxh += maxh[j];
        summaxh2 += maxh2[j];
    }

    Proba &Hi = g[i].H;
    Hi *= rein;
    Hi += g[i].extH;
    real_t eps = 0;
    j = 0;
    for (tie(eit, eend) = out_edges(i, g); eit != eend; ++eit, ++j) {
        real_t const csummaxh = summaxh - maxh[j];
        Proba *U = out.H;

        // case ti = 0
        U[0][0] = csummaxh - mu;
        U[1][0] = -inf;

        // case 0 < ti <= d
        for (int ti = 1; ti < depth; ++ti) {
            U[0][ti] = C[ti].cavityval(j) - ti * rho;
            U[1][ti] = C[ti].cavitymax(j) - ti * rho;
        }

        U[0][depth] = summaxh2 - maxh2[j] - 1;
        U[1][depth] = summaxh2 - maxh2[j] - 1;

        U[0] += Hi;
        U[1] += Hi;
        out.reduce();

        Mes &old = Hin[j];
        eps = l8dist(old, out);
        setMes(*eit, out);
    }

    Ui[0] = summaxh - mu;
    for (int ti = 1; ti < depth; ++ti)
        Ui[ti] = C[ti].fullmax() - ti * rho;
    Ui[depth] = summaxh2 - 1;


    Hi += Ui;

    return eps;
}


void converge(bool output = true) {
    int ite = 0;
    real_t err = 0.0;
    int dec_ite = 0;
    vector<int> permutation(num_vertices(g));
    for (unsigned i = 0; i < num_vertices(g); ++i)
        permutation[i] = i;
    do {
        random_shuffle(permutation.begin(), permutation.end());
        rein = beta_param * ite;
        err = 0;
        unsigned ng = num_vertices(g);

#pragma omp parallel for schedule(dynamic), reduction(max:err)
        for (unsigned i = 0; i < ng; ++i) {
            real_t diff = update(permutation[i]);
            if (diff > err)
                err = diff;
        }

        ++dec_ite;
        int numon2 = 0;
        for (unsigned i = 0; i < num_vertices(g); ++i) {
            Proba &Hi = g[i].H;
            int ti = max_element(&Hi[0], &Hi[0] + depth + 1) - &Hi[0];
            double Hmax = Hi[ti];
            for (int t = 0; t < int(Hi.size()); ++t) {
                Hi[t] -= Hmax;;
            }
            numon2 += (ti < depth);

            if (ti != g[i].t)
                dec_ite = 0;
            g[i].t = ti;
        }
        check_type ck = check_v();
        if (ck.num_bad)
            dec_ite = 0;
#if 1
        cerr << "IT: " << setw(8) << ite << "/" << maxit
             << " | DEC: " << setw(5) << dec_ite << "/" << maxdec
             << " | ERR: " << setw(12) << err << "/" << tolerance
             << " | ON: " << setw(5) << ck.num_on
             << " | S: " << setw(5) << ck.num_seeds
             << " | ON2: " << setw(5) << numon2
             << " | BAD: " << setw(5) << ck.num_bad
             << " | COST: " << setw(12) << ck.tot_cost
             << " | EN: " << setw(12) << ck.tot_energy
             << "   \n"
             << flush;
#endif
    } while (err > tolerance && dec_ite < maxdec && ++ite < maxit);

    if (plotting)
        propagate();
    if (output)
        cerr << endl;
    else
        cerr << string(180, ' ') << '\r' << flush;
}


void leaf_removal(SGraph &sg) {
    vector<int> queue;
    vector<int> depth(num_vertices(sg));
    for (unsigned i = 0; i < num_vertices(sg); ++i) {
        if (out_degree(i, sg) == 1) {
            queue.push_back(i);
            depth[i] = 1;
        }
    }
    int maxd = 0;
    while (!queue.empty()) {
        int i = queue.back();
        maxd = max(maxd, depth[i]);
        queue.resize(queue.size() - 1);
        if (out_degree(i, sg)) {
            int j = target(*(out_edges(i, sg).first), sg);
            clear_vertex(i, sg);
            //cout << "S " << i << endl;
            if (out_degree(j, sg) == 1) {
                depth[j] = depth[i] + 1;
                queue.push_back(j);
            }
        }
    }
    cerr << "# 2core: " << num_edges(sg) << " depth: " << maxd << " edges, " << num_vertices(sg) << " vertices" << endl;
}

void read_graph(istream &file) {
    SGraph sg;

    string tok, tok2;

    while (file >> tok) {
        if (tok == "D" || tok == "E") {
            SVertex i, j;
            file >> i >> j;
            add_edge(i, j, sg);
        } else if (tok == "F") {
            float f;
            Vertex i;
            file >> i >> f;
            if (i >= num_vertices(g)) {
                add_edge(i, 0, g);
                remove_edge(i, 0, g);
            }
            g[i].extH[0] = f;
        } else {
            cout << "token " << tok << " unknown" << endl;
            assert(0);
        }
    }
    cerr << "# graph: " << num_edges(sg) << " edges, " << num_vertices(sg) << " vertices" << endl;

    leaf_removal(sg);


    for (Vertex i = 0; i < num_vertices(g); ++i) {
        for (int t = 0; t <= depth; ++t)
            g[i].extH[t] += real01() * noise;
    }
    for (SVertex i = 0; i < num_vertices(sg); ++i) {
        sedge_iterator eit, end;
        for (tie(eit, end) = out_edges(i, sg); eit != end; ++eit) {
            SVertex j = target(*eit, sg);
            if (i < j)
                add_edge(i, j, g);
        }
    }

}

namespace po = boost::program_options;

po::variables_map parse_command_line(int ac, char **av) {
    po::options_description desc("Usage: " + string(av[0]) + " <option> ... \n\twhere <option> is one or more of");
    desc.add_options()
            ("help", "produce help message")
            ("depth,d", po::value(&depth)->default_value(depth),
             "set maximum time depth")
            ("maxit,t", po::value(&maxit)->default_value(maxit),
             "set maximum number of iterations")
            ("macdec,D", po::value(&maxdec)->default_value(maxdec),
             "set maximum number of decisional iterations")
            ("tolerance,e", po::value(&tolerance)->default_value(tolerance),
             "set convergence tolerance")
            ("rein,g", po::value<real_t>(&beta_param)->default_value(beta_param),
             "sets reinforcement parameter rein")
            ("mu,m", po::value<real_t>(&mu)->default_value(mu), "sets mu parameter")
            ("rho,R", po::value<real_t>(&rho)->default_value(rho), "sets time damping")
            ("noise,r", po::value<real_t>(&noise)->default_value(noise), "sets noise")
            ("seed,s", po::value<unsigned>(), "sets instance seed")
            ("mseed,z", po::value<unsigned>(), "sets messages seed")
            ("output,o", "outputs optimal seeds to std output")
            ("fields,F", "output fields on convergence")
            ("times,T", "output times on convergence")
            ("plotting,P", "output times while converging")
#ifdef OMP
        ("cpu,j", po::value(&numthreads)->default_value(numthreads), "number of cpu")
#endif
            ;

    po::variables_map vm;
    po::store(po::parse_command_line(ac, av, desc), vm);
    po::notify(vm);


    if (vm.count("seed")) {
        unsigned s = vm["seed"].as<unsigned>();
        gen.seed(s);
    }
    if (vm.count("mseed")) {
        unsigned s = vm["mseed"].as<unsigned>();
        mes_gen.seed(s);
    }

    if (vm.count("help")) {
        cerr << desc << "\n";
        exit(1);
    }

    omp_set_num_threads(numthreads);
    Mem.resize(numthreads);

    return vm;
}


int main(int ac, char **av) {
    cout.setf(ios_base::fixed, ios_base::floatfield);
    po::variables_map vm = parse_command_line(ac, av);

    read_graph(cin);
//	cout << "# Parameters: depth=" << depth
//		<< "  beta_param=" << beta_param
//		<< "  mu=" << mu
//		<< "  maxit=" << maxit
//		<< "  maxdec=" << maxdec
//		<< "  tolerance=" << tolerance << endl;

    if (vm.count("plotting"))
        plotting = true;

    converge();

    check_v(true);

    if (vm.count("output"))
        print_seeds();
    if (vm.count("times"))
        print_times();
    if (vm.count("fields"))
        print_fields();

    return 0;
}


