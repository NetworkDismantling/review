/*
********************************************************************************
TAbyTwoCoreV01.cpp

STATEMENT:
This code is aimed at approximately solving an important problem of network
science, namely to divide a network into small components by deleting vertices
in a most economic way.

DESCRIPTION:

This program is applicable on a single graph instance. The imput graph file has
the following form:

N   M                % first row specifies N (vertex number) and M (edge number)

i_1  j_1                                            % undirected edge (i_1, j_1)
i_2  j_2                                            % undirected edge (i_2, j_2)
.    .
.    .
.    .
i_M  j_M                                            % undirected edge (i_M, j_M)

The program reads only the first EdgeNumber edges from the imput graph, with
EdgeNumber being explicitly specified (EdgeNumber <= M, of course).

Each vertex has two states (unoccupied, 0; occupied, 1). An unoccupied vertex
belongs to the constructed target set S, while an occupied vertex remains in the
network.

The target set S is contructed by three steps:
1) a feedback vertex set S0 is constructed.
2) if a tree component of the remaining subgraph contains more than Sthreshold
vertices, one of its vertices is deleted.
3) some deleted vertices are added back to the remaining network as long as
the netowrk component sizes do not exceed Sthreshold.

LOG:
23.06.2016 - 24.06.2016: revision of TAbyTwoCoreV01.cpp.
23.06.2016: copied from TAbyFVSbpdV04.cpp to TAbyTwoCoreV01.cpp.

PROGRAMMER:
Hai-Jun Zhou
Institute of Theoretical Physics, Chinese Academy of Sciences
Zhong-Guan-Cun East Road 55, Beijing 100190
email: zhouhj@itp.ac.cn
webpage: power.itp.ac.cn/~zhouhj/
********************************************************************************
*/

#include <exception>
#include <fstream>
#include <iostream>
#include <queue>
#include <set>
#include <stack>
#include <string>
#include <valarray>
#include <ctime>
#include <boost/program_options.hpp>

#include "zhjrandom.h" //a random number generator


using namespace boost;
using namespace std;

namespace po = boost::program_options;

namespace params {
    int VertexNumber;
    int EdgeNumber;
    // You can set Csize to be 1 percent of VertexNumber of even smaller
    // assumed to be 1 percent
    int Csize;
    string graphfile;
    string Afile;
    string FVSfile;
    string Timefile;

    //  random number generator initialization
    int rdseed = 93276792; //you can set this seed to another value
    int prerun = 14000000; //you can set it to another value

}

using namespace params;

/*---               random real uniformly distributed in [0,1)            ---*/
double u01prn(ZHJRANDOMv3 *rd) {
    return rd->rdflt();
}

struct IntPair {
    int first;
    int second;

    IntPair();

    IntPair(int, int);
};

IntPair::IntPair() {
    first = 0;
    second = 0;
}

IntPair::IntPair(int a, int b) {
    first = a;
    second = b;
}

bool operator<(IntPair a, IntPair b) {
    if (a.first < b.first)
        return true;
    else if (a.first > b.first)
        return false;
    else //a.first = b.first
    {
        if (a.second < b.second)
            return true;
        else
            return false;
    }
}

bool operator==(IntPair a, IntPair b) {
    return (a.first == b.first) && (a.second == b.second);
}

struct outmessage //cavity out message to a vertex
{
    struct vstruct *v_ptr; //pointer to the receiving vertex
    outmessage();
};

outmessage::outmessage() {
    v_ptr = nullptr;
}

struct vstruct //vertex struct
{
    int index;                 //index of vertex, positive integer
    int degree;                //number of neighboring vertices
    int active_degree;         //number of active neighboring vertices
    int c_index;               //index of component
    bool occupied;             //=true (not deleted); =false (deleted)
    bool active;               //=true (need to be considered in BP iteration)
    struct outmessage *om_ptr; //start position of output message
    int b_size;                //size of a branch (used in ComponentRefinement)
    vstruct();
};

vstruct::vstruct() {
    index = 0;
    degree = 0;
    active_degree = 0;
    c_index = 0;
    occupied = true; //initially all vertices are occupied
    active = true;   //initially all vertices are active
    om_ptr = nullptr;
    b_size = 0;
}

class FVS //feedback vertex set
{
public:
    explicit FVS(ZHJRANDOMv3 *);                      //constructor
    bool Graph(string &, int);               //read graph connection pattern
    int Fix0();                          //fix some variables to be un-occupied and simplify
    void ComponentRefinement(int, string &); //refinement of components
    void AttackEffect(string &);             //the accumulated effect of node deletion
//    bool ReadFVS(string &);                  //read a FVS into the program
    void Simplify(struct vstruct *);         //simplify the graph by removing leaf

private:
    int VertexNumber;        //total number of vertices
    int ActiveVertexNumber;  //total number of active vertices
    int ActiveVertexNumber0; //total # of active vertices before this BPD round
    int EdgeNumber;          //total number of edges

    ZHJRANDOMv3 *PRNG; //random number generator

    stack<int> Targets; //target attack nodes

    valarray<vstruct> Vertex;
    valarray<outmessage> OutMessage;

    valarray<int> CandidateVertices; //list of candidate vertices to be fixed
    valarray<int> Permutation;       //array used in random sequential updating
    int CandidateNumber;             //number of vertices in CandidateVertices
};


po::variables_map parse_command_line(int ac, char **av) {
    po::options_description desc("Usage: " + string(av[0]) + " <option> ... \n\twhere <option> is one or more of");

    desc.add_options()
            ("help", "produce help message")
            ("NetworkFile,F", po::value(&graphfile), "File containing the network")
            ("VertexNumber,N", po::value(&VertexNumber), "Number of vertices of the network")
            ("EdgeNumber,E", po::value(&EdgeNumber), "Number of edges of the network")
            ("Csize,C", po::value(&Csize), "Target component size")
            ("seed,s", po::value(&rdseed)->default_value(rdseed), "Pseudo-random number generator seed")
            ("prerun,P", po::value(&prerun)->default_value(prerun), "Pre-run number of iterations")
            ("Afile,A", po::value(&Afile), "File containing the adjacency matrix")
            ("FVSfile,O", po::value(&FVSfile), "[UNUSED] File containing the Feedback Vertex Set (FVS)")
            ("Timefile,T", po::value(&Timefile), "File containing the time to find the feedback");

    po::variables_map vm;
    po::store(po::parse_command_line(ac, av, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        exit(1);
    }

    return vm;
}


/*---                       constructor of FVS cluster                    ---*/
FVS::FVS(ZHJRANDOMv3 *rd) {
    PRNG = rd; //random number generator
}

/* -                           Read graph from file                           -
gname: the input graph name.
enumber: read the first enumber edges only.
*/
bool FVS::Graph(string &gname, int enumber) {
    ifstream graphf(gname.c_str());
    if (!graphf.good()) {
        cerr << "Graph probably non-existent.\n";
        return false;
    }
    while (!Targets.empty()) {
        Targets.pop();
    }
    //first read of input graph
    graphf >> VertexNumber >> EdgeNumber;
    if (EdgeNumber < enumber) {
        cerr << "No so many edges in the graph (got: " << EdgeNumber << ", expected: " << enumber << ").\n";
        graphf.close();
        return false;
    }
    EdgeNumber = enumber; //only the first enum edges are read into
    try {
        Vertex.resize(VertexNumber + 1);
    }
    catch (std::bad_alloc const &) {
        cerr << "Vertex construction failed.\n";
        graphf.close();
        return false;
    }
    try {
        Permutation.resize(VertexNumber + 1);
    }
    catch (std::bad_alloc const &) {
        cerr << "Permutation construction failed.\n";
        return false;
    }
    try {
        CandidateVertices.resize(VertexNumber + 1);
    }
    catch (std::bad_alloc const &) {
        cerr << "CandidateVertices construction failed.\n";
        return false;
    }
    bool succeed = true;
    set<IntPair> EdgeSet;
    for (int eindex = 0; eindex < EdgeNumber && succeed; ++eindex) {
        int v1, v2;
        graphf >> v1 >> v2;
        if (v1 > v2) {
            int v3 = v1;
            v1 = v2;
            v2 = v3;
        }
        if (v1 == v2 || v1 == 0 || v1 > VertexNumber || v2 == 0 || v2 > VertexNumber) {
            cerr << "Graph incorrect at line " << eindex + 1 << endl;
            succeed = false;
        } else if (EdgeSet.find(IntPair(v1, v2)) != EdgeSet.end()) {
            cerr << "Multiple edges.\n";
            succeed = false;
        } else {
            EdgeSet.insert(IntPair(v1, v2));
            ++(Vertex[v1].degree);
            ++(Vertex[v2].degree);
        }
    }
    graphf.close();
    if (!succeed)
        return false;
    EdgeSet.clear();
    try {
        OutMessage.resize(2 * EdgeNumber);
    }
    catch (std::bad_alloc const &) {
        cerr << "OutMessage construction failed.\n";
        return false;
    }
    int position = 0;
    struct vstruct *v_ptr = &Vertex[1];
    for (int v = 1; v <= VertexNumber; ++v, ++v_ptr) {
        v_ptr->index = v;
        v_ptr->occupied = true;
        v_ptr->active = true;
        v_ptr->active_degree = v_ptr->degree;
        v_ptr->om_ptr = &OutMessage[position];
        position += v_ptr->degree;
        v_ptr->degree = 0;
    }
    //second read of input graph
    graphf.open(gname.c_str());
    graphf >> VertexNumber >> EdgeNumber;
    EdgeNumber = enumber;
    struct outmessage *om_ptr = &OutMessage[0];
    for (int eindex = 0; eindex < EdgeNumber; ++eindex) {
        int v1, v2;
        graphf >> v1 >> v2;
        if (v1 > v2) {
            int v3 = v1;
            v1 = v2;
            v2 = v3;
        }
        om_ptr = Vertex[v1].om_ptr + Vertex[v1].degree;
        om_ptr->v_ptr = &Vertex[v2];
        om_ptr = Vertex[v2].om_ptr + Vertex[v2].degree;
        om_ptr->v_ptr = &Vertex[v1];
        ++(Vertex[v1].degree);
        ++(Vertex[v2].degree);
    }
    graphf.close();
    cout << "Graph: N= " << VertexNumber
         << ",  M= " << EdgeNumber
         << ",  Mean degree= " << (2.0 * EdgeNumber) / (1.0 * VertexNumber) << endl;
    cout.flush();
    ActiveVertexNumber = VertexNumber;
    v_ptr = &Vertex[1];
    for (int v = 1; v <= VertexNumber; ++v, ++v_ptr)
        if (v_ptr->active && v_ptr->active_degree <= 1) {
            v_ptr->active = false;
            v_ptr->occupied = true; //being occupied
            --ActiveVertexNumber;
            Simplify(v_ptr);
        }
    if (ActiveVertexNumber == 0) {
        cout << "The graph has no loops. Done! \n";
        cout.flush();
    }
    v_ptr = &Vertex[1];
    ActiveVertexNumber = 0;
    for (int v = 1; v <= VertexNumber; ++v, ++v_ptr)
        if (v_ptr->active) {
            Permutation[ActiveVertexNumber] = v_ptr->index;
            ++ActiveVertexNumber;
        }
    ActiveVertexNumber0 = ActiveVertexNumber;
    CandidateNumber = 0;
    return true;
}

/*                simplify after fixing variable                             */
void FVS::Simplify(struct vstruct *v_ptr) {
    struct outmessage *om_ptr = v_ptr->om_ptr;
    for (int d = 0; d < v_ptr->degree; ++d, ++om_ptr) {
        if (--(om_ptr->v_ptr->active_degree) <= 1) {
            if (om_ptr->v_ptr->active) {
                om_ptr->v_ptr->active = false;
                om_ptr->v_ptr->occupied = true;
                --ActiveVertexNumber;
                Simplify(om_ptr->v_ptr);
            }
        }
    }
}

/*-- externally fixing one vertex to be empty and simplify the system --*/
int FVS::Fix0() {
    struct vstruct *v_ptr = nullptr;
    int DeletionNumber = 0;
    int MaxActiveDegree = 0;
    CandidateNumber = 0;
    while (ActiveVertexNumber > 0) {
        if (CandidateNumber == 0) {
            MaxActiveDegree = 0;
            for (int quant = ActiveVertexNumber0 - 1; quant >= 0; --quant) {
                if (!Vertex[Permutation[quant]].active) {
                    --ActiveVertexNumber0;
                    Permutation[quant] = Permutation[ActiveVertexNumber0];
                } else {
                    v_ptr = &Vertex[Permutation[quant]];
                    if (v_ptr->active_degree > MaxActiveDegree) {
                        MaxActiveDegree = v_ptr->active_degree;
                        CandidateVertices[0] = v_ptr->index;
                        CandidateNumber = 1;
                    } else if (v_ptr->active_degree == MaxActiveDegree) {
                        CandidateVertices[CandidateNumber] = v_ptr->index;
                        ++CandidateNumber;
                    }
                }
            }
        } else {
            int index = static_cast<int>(CandidateNumber * u01prn(PRNG));
            v_ptr = &Vertex[CandidateVertices[index]];
            --CandidateNumber;
            CandidateVertices[index] = CandidateVertices[CandidateNumber];
            if (v_ptr->active && v_ptr->active_degree == MaxActiveDegree) {
                v_ptr->active = false;
                v_ptr->occupied = false;
                ++DeletionNumber;
                Targets.push(v_ptr->index);
                --ActiveVertexNumber;
                Simplify(v_ptr);
            }
        }
    }
    cout << DeletionNumber << endl;
    return DeletionNumber;
}

/*
Refinement of the components.
Sthreshold is the maximal allowed component size.
*/
void FVS::ComponentRefinement(int Sthreshold, string &outfile) {
    int num_empty = 0;
    struct vstruct *v_ptr = &Vertex[1];
    for (int v = 1; v <= VertexNumber; ++v, ++v_ptr) {
        v_ptr->c_index = 0;
        if (!v_ptr->occupied)
            ++num_empty;
    }
    int max_comp_size = 0;

    //           determine the size of each tree component, and break giant trees
    int c_index = 0;      //index of component
    int c_size = 0;       //size of component
    int NumberDelete = 0; //number of deleted vertices to break giant trees
    // Notice! In this subroutine, Permutation stores the size of each component.
    Permutation = 0;
    v_ptr = &Vertex[1];
    struct outmessage *om_ptr;
    for (int v = 1; v <= VertexNumber; ++v, ++v_ptr)
        if (v_ptr->occupied && v_ptr->c_index == 0) {
            ++c_index;
            c_size = 1;
            Vertex[v_ptr->index].c_index = c_index;
            queue<int> members;
            members.push(v_ptr->index);
            while (!members.empty()) {
                int vtx = members.front();
                members.pop();
                om_ptr = Vertex[vtx].om_ptr;
                for (int d = 0; d < Vertex[vtx].degree; ++d, ++om_ptr)
                    if (om_ptr->v_ptr->occupied) {
                        int vtx2 = om_ptr->v_ptr->index;
                        if (Vertex[vtx2].c_index == 0) {
                            Vertex[vtx2].c_index = c_index;
                            members.push(vtx2);
                            c_size += 1;
                        }
                    }
            }
            if (c_size <= Sthreshold) {
                Permutation[c_index] = c_size;
                if (c_size > max_comp_size)
                    max_comp_size = c_size;
            } else {
                /* break giant tree component. Choose a vertex 'cvtx' such that its
                   deletion decreases the tree component size maximally. */
                int cvtx;                      // vertex to be deleted
                double min_max_csize = c_size; //minimal possible size of max-sub-tree
                queue<int> leafnodes;
                set<int> gtree;
                int vtx = v_ptr->index;
                gtree.insert(vtx);
                members.push(vtx);
                while (!members.empty()) {
                    vtx = members.front();
                    members.pop();
                    Vertex[vtx].active = true;
                    Vertex[vtx].active_degree = 0;
                    om_ptr = Vertex[vtx].om_ptr;
                    for (int d = 0; d < Vertex[vtx].degree; ++d, ++om_ptr)
                        if (om_ptr->v_ptr->c_index == c_index) {
                            Vertex[vtx].active_degree += 1;
                            int vtx2 = om_ptr->v_ptr->index;
                            if (gtree.find(vtx2) == gtree.end()) {
                                members.push(vtx2);
                                gtree.insert(vtx2);
                            }
                        }
                    if (Vertex[vtx].active_degree == 1)
                        leafnodes.push(vtx);
                }
                while (!leafnodes.empty()) {
                    vtx = leafnodes.front();
                    leafnodes.pop();
                    Vertex[vtx].active = false;
                    int maxbsize = 0; //maximal branch size
                    int psum = 0;
                    int d0 = -1; //position of active neighbor
                    om_ptr = Vertex[vtx].om_ptr;
                    for (int d = 0; d < Vertex[vtx].degree; ++d, ++om_ptr)
                        if (om_ptr->v_ptr->occupied) {
                            om_ptr->v_ptr->active_degree -= 1;
                            if (!om_ptr->v_ptr->active) {
                                int bsize = om_ptr->v_ptr->b_size;
                                if (maxbsize < bsize)
                                    maxbsize = bsize;
                                psum += bsize;
                            } else {
                                d0 = d;
                                if (om_ptr->v_ptr->active_degree == 1)
                                    leafnodes.push(om_ptr->v_ptr->index);
                            }
                        }
                    if (d0 >= 0) {
                        int bsize = c_size - 1 - psum;
                        if (maxbsize < bsize)
                            maxbsize = bsize;
                        Vertex[vtx].b_size = psum + 1;
                    }
                    if (maxbsize < min_max_csize) {
                        cvtx = vtx;
                        min_max_csize = maxbsize;
                    }
                }
                Vertex[cvtx].occupied = false; //vertex deleted
                ++num_empty;
                Targets.push(cvtx);
                ++NumberDelete;
                for (int sci: gtree) {
                    Vertex[sci].c_index = 0;
                }
                --c_index;
                --v;
                --v_ptr;
            }
        }
    //        add some vertices to small components and/or merge small components
    max_comp_size = 0;
    int NumberAddition = 0;
    set<IntPair> candidates;
    v_ptr = &Vertex[1];
    for (int v = 1; v <= VertexNumber; ++v, ++v_ptr)
        if (v_ptr->c_index == 0) {
            set<int> cgroups;
            int vtx = v_ptr->index;
            om_ptr = v_ptr->om_ptr;
            for (int d = 0; d < v_ptr->degree; ++d, ++om_ptr) {
                int vtx2 = om_ptr->v_ptr->index;
                if (Vertex[vtx2].c_index != 0)
                    cgroups.insert(Vertex[vtx2].c_index);
            }
            c_size = 1;
            for (int cgroup: cgroups)
                c_size += Permutation[cgroup];
            if (c_size <= Sthreshold)
                candidates.insert(IntPair(c_size, vtx));
        }
    while (!candidates.empty()) {
        c_index = 0;
        auto sci = candidates.begin();
        int csize0 = sci->first;
        int vtx = sci->second;
        candidates.erase(IntPair(csize0, vtx));
        set<int> cgroups;
        om_ptr = Vertex[vtx].om_ptr;
        for (int d = 0; d < Vertex[vtx].degree; ++d, ++om_ptr) {
            int vtx2 = om_ptr->v_ptr->index;
            if (Vertex[vtx2].c_index != 0) {
                cgroups.insert(Vertex[vtx2].c_index);
                if (c_index < Vertex[vtx2].c_index)
                    c_index = Vertex[vtx2].c_index;
            }
        }
        c_size = 1;
        for (int cgroup: cgroups)
            c_size += Permutation[cgroup];
        if (c_size <= Sthreshold) {
            if (c_size > csize0)
                candidates.insert(IntPair(c_size, vtx));
            else {
                ++NumberAddition;
                Vertex[vtx].occupied = true;
                --num_empty;
                Permutation[c_index] = c_size;
                if (max_comp_size < c_size)
                    max_comp_size = c_size;
                queue<int> members;
                members.push(vtx);
                Vertex[vtx].c_index = c_index;
                while (!members.empty()) {
                    int vtx2 = members.front();
                    members.pop();
                    om_ptr = Vertex[vtx2].om_ptr;
                    for (int d = 0; d < Vertex[vtx2].degree; ++d, ++om_ptr) {
                        int vtx3 = om_ptr->v_ptr->index;
                        if (Vertex[vtx3].c_index != c_index &&
                            Vertex[vtx3].occupied) {
                            Permutation[Vertex[vtx3].c_index] = 0;
                            Vertex[vtx3].c_index = c_index;
                            members.push(vtx3);
                        }
                    }
                }
            }
        }
    }
    ofstream pfile(outfile.c_str());
    pfile << "Targets  " << num_empty << endl
          << endl;
    v_ptr = &Vertex[1];
    for (int v = 1; v <= VertexNumber; ++v, ++v_ptr)
        if (!v_ptr->occupied)
            pfile << v_ptr->index << endl;
    pfile.close();
}

/* Report the attack order and its effect. */
void FVS::AttackEffect(string &attackfile) {
    struct vstruct *v_ptr = &Vertex[1];
    for (int v = 1; v <= VertexNumber; ++v, ++v_ptr)
        v_ptr->c_index = 0;
    set<int> finaltargets;
    int max_comp_size = 0;
    int num_empty = 0;
    //                                       determine the size of each component
    int c_index = 0; //index of component
    int c_size = 0;  //size of component
    //  Notice! In this subroutine, Permutation stores the size of each component
    Permutation = 0;
    v_ptr = &Vertex[1];
    struct outmessage *om_ptr;
    for (int v = 1; v <= VertexNumber; ++v, ++v_ptr) {
        if (!v_ptr->occupied) {
            ++num_empty;
            finaltargets.insert(v_ptr->index);
        } else if (v_ptr->c_index == 0) {
            ++c_index;
            c_size = 1;
            Vertex[v_ptr->index].c_index = c_index;
            queue<int> members;
            members.push(v_ptr->index);
            while (!members.empty()) {
                int vtx = members.front();
                members.pop();
                om_ptr = Vertex[vtx].om_ptr;
                for (int d = 0; d < Vertex[vtx].degree; ++d, ++om_ptr)
                    if (om_ptr->v_ptr->occupied) {
                        int vtx2 = om_ptr->v_ptr->index;
                        if (Vertex[vtx2].c_index == 0) {
                            Vertex[vtx2].c_index = c_index;
                            members.push(vtx2);
                            c_size += 1;
                        }
                    }
            }
            Permutation[c_index] = c_size;
            if (c_size > max_comp_size)
                max_comp_size = c_size;
        }
    }
    ofstream output(attackfile.c_str());
    output << (1.0e0 * num_empty) / (1.0e0 * VertexNumber) << '\t'
           << (1.0e0 * max_comp_size) / (1.0e0 * VertexNumber) << endl;
    while (!Targets.empty()) {
        int vtx = Targets.top();
        Targets.pop();
        if (!Vertex[vtx].occupied) {
            int c_index = 0;
            set<int> cgroups;
            om_ptr = Vertex[vtx].om_ptr;
            for (int d = 0; d < Vertex[vtx].degree; ++d, ++om_ptr) {
                int vtx2 = om_ptr->v_ptr->index;
                if (Vertex[vtx2].c_index != 0) {
                    cgroups.insert(Vertex[vtx2].c_index);
                    if (c_index < Vertex[vtx2].c_index)
                        c_index = Vertex[vtx2].c_index;
                }
            }
            int c_size = 1;
            for (int cgroup: cgroups)
                c_size += Permutation[cgroup];
            Vertex[vtx].occupied = true;
            --num_empty;
            Permutation[c_index] = c_size;
            if (max_comp_size < c_size)
                max_comp_size = c_size;
            queue<int> members;
            members.push(vtx);
            Vertex[vtx].c_index = c_index;
            while (!members.empty()) {
                int vtx2 = members.front();
                members.pop();
                om_ptr = Vertex[vtx2].om_ptr;
                for (int d = 0; d < Vertex[vtx2].degree; ++d, ++om_ptr) {
                    int vtx3 = om_ptr->v_ptr->index;
                    if (Vertex[vtx3].c_index != c_index &&
                        Vertex[vtx3].occupied) {
                        Permutation[Vertex[vtx3].c_index] = 0;
                        Vertex[vtx3].c_index = c_index;
                        members.push(vtx3);
                    }
                }
            }
            output << (1.0e0 * num_empty) / (1.0e0 * VertexNumber) << '\t'
                   << (1.0e0 * max_comp_size) / (1.0e0 * VertexNumber) << endl;
        }
    }
    output.close();
    for (int finaltarget: finaltargets)
        Vertex[finaltarget].occupied = false;
}


int main(int argc, char **argv) {
    po::variables_map vm = parse_command_line(argc, argv);

    cout << "NetworkFile = " << graphfile << endl;

    auto rdgenerator = new ZHJRANDOMv3(rdseed);
    for (int i = 0; i < prerun; ++i)
        rdgenerator->rdflt();

    auto system = new FVS(rdgenerator);

    bool succeed = system->Graph(graphfile, EdgeNumber);
    if (!succeed) return -1;

    clock_t t1c = clock();
    time_t t1 = time(nullptr);

    int DeletionNumber = system->Fix0();
    cout << "FVS size = " << DeletionNumber << endl;

    //          report feedback vertex set, please change the name as you prefer
    /*
    if( system.CheckFVS(FVSfile) == false)
    {
      cerr<<"Not a feedback vertex set.\n";
      return -1;
    }
    */

    system->ComponentRefinement(Csize, Afile);

    clock_t t2c = clock();
    time_t t2 = time(nullptr);

    ofstream Toutf(Timefile.c_str());
    Toutf << "Total seconds used="
          << static_cast<double>(t2c - t1c) / CLOCKS_PER_SEC
          << " s" << endl;
    Toutf << "Total time used = " << t2 - t1 << "s" << endl;
    Toutf.close();

    //The following two lines report size evolution of the largest component
    //  string Bfile="ERn100kM1mg100.TAeffect";
    //  system.AttackEffect(Bfile);

    //  return 1;
    return 0;
}
