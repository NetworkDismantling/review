#include <iostream>
#include <fstream>
#include <forward_list>
#include <list>

#include <vector>
#include <ranges>

#include <Python.h>

#define GRAPH_TOOL

#ifdef GRAPH_TOOL
#include <graph.hh>
#include <graph_python_interface.hh>
#endif

#ifdef BOOST
#include <boost/python.hpp>
#include <utility>
#endif

#include "tsl/robin_map.h"

using namespace std;
using namespace tsl;

#if defined(GRAPH_TOOL)

// Define namespace graph_tool as gt
namespace gt = graph_tool;
using NodeType = gt::GraphInterface::multigraph_t::vertex_t;

#else

using NodeType = size_t;

#endif


// Type definitions
using ComponentType = unsigned int;

class Graph {
private:
    // Graph representation (adjacency list)
    robin_map<NodeType, forward_list<NodeType> > g;
    robin_map<NodeType, bool> vis;
    vector<NodeType> stack;
    robin_map<ComponentType, vector<NodeType> *> component;
    //    robin_map<int, vector<unsigned int> *> degrees;
    robin_map<NodeType, ComponentType> node2comp;
    long int counterCC;
    string name;
    bool first = true;
    bool directed = false;

    // Private initialization methods
#ifdef GRAPH_TOOL
    void initFromGraphInterface(gt::GraphInterface &graphInterface) {
        directed = graphInterface.get_directed();
        
        auto const &adj_list = graphInterface.get_graph();
        g = robin_map<NodeType, forward_list<NodeType>>();
        g.reserve(graphInterface.get_num_vertices());

        for (auto [src_it, end] =  vertices(adj_list); src_it != end; ++src_it) {
            auto const srcNode = *src_it;

            g[srcNode] = forward_list<NodeType>();
            for (auto [dst_it, nend] = adjacent_vertices(srcNode, adj_list); dst_it != nend; ++dst_it) {
                auto const dstNode = *dst_it;
                g[srcNode].push_front(dstNode);

                if (graphInterface.get_directed() == false) {
                    g[dstNode].push_front(srcNode);
                }
            }
        }
    }
#endif

#ifdef BOOST
    void initFromEdgeList(const boost::python::list &edgelist) {
        addEdgeList(edgelist);
    }
#endif

public:
    // Default constructor needed by Boost.Python
    [[nodiscard]]
    Graph() : Graph("") {

    }

    [[nodiscard]]
    explicit Graph(string n) : counterCC(-1), name(std::move(n)) {
        this->first = true;
    }

    [[nodiscard]]
    // Copy constructor
    explicit Graph(const Graph *graph) : Graph(graph->name) {
        // g = graph->g;

        first = graph->first;
        directed = graph->directed;
        
        // Deep copy the adjacency list
        for (const auto &[node, neighbors] : graph->g) {
            g[node] = forward_list<NodeType>();
            // Copy all neighbors
            for (const auto &neighbor : neighbors) {
                g[node].push_front(neighbor);
            }
        }
    }
    


#ifdef GRAPH_TOOL
    // Constructor from GraphInterface (internal C++ interface)
    [[nodiscard]]
    explicit
    Graph(gt::GraphInterface &graphInterface) : Graph("") {
        initFromGraphInterface(graphInterface);
    }
#endif

#ifdef BOOST
    // Universal constructor from Python object with type checking
    // Accepts: string (graph name), list (edge list), or graph_tool.Graph
    [[nodiscard]]
    explicit
    Graph(const boost::python::object& obj) : Graph("") {
        // Try string first (graph name)
        boost::python::extract<string> stringExtractor(obj);
        if (stringExtractor.check()) {
            name = stringExtractor();
            this->first = true;
            return;
        }
        
        // Try list (edge list)
        boost::python::extract<boost::python::list> listExtractor(obj);
        if (listExtractor.check()) {
            boost::python::list edgeList = listExtractor();
            initFromEdgeList(edgeList);
            return;
        }
        
#ifdef GRAPH_TOOL
        // Try graph_tool.Graph (extract internal GraphInterface)
        try {
            boost::python::object graphInterface = obj.attr("_Graph__graph");
            gt::GraphInterface& gi = boost::python::extract<gt::GraphInterface&>(graphInterface);
            initFromGraphInterface(gi);
            return;
        } catch (boost::python::error_already_set&) {
            PyErr_Clear();
        }
#endif
        
        // If nothing worked, raise an error
        PyErr_SetString(PyExc_TypeError, 
            "Graph constructor expects: string (graph name), list (edge list), or graph_tool.Graph");
        boost::python::throw_error_already_set();
    }
    
    explicit Graph(const boost::python::list& pythonList): Graph("") {
        initFromEdgeList(pythonList);
    }
#endif

    ~Graph() {
        // robin_map<int, vector<int> *>::iterator itc, cend;

        for (auto &[fst, snd]: component) {
            delete snd;
        }
        //        for (itc = component.begin(), cend = component.end(); itc != cend; ++itc) {
        //            delete itc->second;
        //        }
    }

    // Deep copy method - creates a new Graph with copied adjacency list
    [[nodiscard]]
    Graph* deepCopy() const {
        return new Graph(this);
        // Graph* copy = new Graph(name);
    
        // copy->first = first;
        // copy->directed = directed;
        
        // // Deep copy the adjacency list
        // for (const auto &[node, neighbors] : g) {
        //     copy->g[node] = forward_list<NodeType>();
        //     // Copy all neighbors
        //     for (const auto &neighbor : neighbors) {
        //         copy->g[node].push_front(neighbor);
        //     }
        // }
        
        // return copy;
    }

    [[nodiscard]]
    bool isEmpty() const {
        return g.empty();
    }

    [[nodiscard]]
    bool getFirst() const {
        return first;
    }

    void setFirst(const bool val) {
        first = val;
    }

    [[nodiscard]]
    unsigned long size() const {
        return g.size();
    }

    [[nodiscard]]
    NodeType getNumNodes() const {
        return g.size();
    }
    [[nodiscard]]
    size_t getNumEdges() const {
        size_t edgeCount = 0;
        for (const auto &[fst, snd]: g) {
            // edgeCount += snd.size();
            edgeCount += std::distance(snd.begin(), snd.end());
        }

        if (directed)
            return edgeCount;
        else
            return edgeCount / 2;
    }

    bool addNode(const unsigned int nodeID) {
        if (g.count(nodeID) > 0) {
            fprintf(stderr, "ERROR: %u already present in the graph: %s\n", nodeID, name.c_str());
            return false;
        } else {
            g[nodeID] = forward_list<NodeType>();
            return true;
        }
    }

    bool addNodes_python(const boost::python::object& obj) {
        // Try list first
        boost::python::extract<boost::python::list> listExtractor(obj);
        if (listExtractor.check()) {
            boost::python::list nodes = listExtractor();
            return addNodes(nodes);
        }

        // Try vector next
        boost::python::extract<vector<NodeType>> vectorExtractor(obj);
        if (vectorExtractor.check()) {
            vector<NodeType> nodes = vectorExtractor();
            return addNodes(nodes);
        }

        // If nothing worked, raise an error
        PyErr_SetString(PyExc_TypeError, 
            "addNodes expects a list or vector of node IDs");
        boost::python::throw_error_already_set();
        return false; // Unreachable, but avoids compiler warning
    }
    bool addNodes(const boost::python::list &nodes) {
        const boost::python::ssize_t nSize = boost::python::len(nodes);

        for (boost::python::ssize_t i = 0; i < nSize; ++i) {
            addNode(boost::python::extract<NodeType>(nodes[i]));
        }
        return true;
    }
    bool addNodes(const vector<NodeType> &nodes) {
        for (const auto &nodeID: nodes) {
            addNode(nodeID);
        }
        return true;
    }

    bool removeNode(const NodeType nodeID) {
        // forward_list<unsigned int>::iterator ite;

        if (g.count(nodeID) == 0) {
            cout << "ERROR: Node " << nodeID << " not present in the graph: " << name << endl;
            return false;
        } else {
            // vec[i] = std::move(vec.back()); vec.pop_back();
            for (auto node: g[nodeID]) {
                g[node].remove(nodeID);
            }
            //            for (ite = g[nodeID].begin(); ite != g[nodeID].end(); ++ite)
            //                g[*ite].remove(nodeID);

            g.erase(nodeID);
            return true;
        }
    }

    bool clearNode(const NodeType nodeID) {
        // Clear all edges from a node but keep the node itself (matches Python clear_vertex)
        if (g.count(nodeID) == 0) {
            cout << "ERROR: Node " << nodeID << " not present in the graph: " << name << endl;
            return false;
        } else {
            // Remove this node from all neighbors' adjacency lists
            for (auto neighbor: g[nodeID]) {
                g[neighbor].remove(nodeID);
            }
            // Clear this node's adjacency list
            g[nodeID].clear();
            return true;
        }
    }

    bool addEdge(const unsigned int srcNode, const unsigned int dstNode) {
        g[srcNode].push_front(dstNode);
        g[dstNode].push_front(srcNode);
        return true;
    }

    bool removeEdge(const unsigned int srcNode, const unsigned int dstNode) {
        g[srcNode].remove(dstNode);
        g[dstNode].remove(srcNode);
        return true;
    }

    bool loadEdgeListFromFile(const string &fname) {
        ifstream inFile;
        int s, d;

        inFile.open(fname);
        if (!inFile) {
            cout << "Unable to open file";
            return false;
        }

        while (inFile >> s) {
            inFile >> d;
            //To cope with selfloops
            if (s == d) continue;
            addEdge(s, d);
        }
        inFile.close();
        return true;
    }

    bool addEdgeList(const vector<pair<NodeType, NodeType> > &el) {
        for (auto &[fst, snd]: el) {
            addEdge(fst, snd);
        }
        return true;
    }

#ifdef BOOST
    bool loadGraphFromNumpyArray() {
        // TODO
        PyErr_SetString(PyExc_NotImplementedError, 
            "loadGraphFromNumpyArray not yet implemented");
        boost::python::throw_error_already_set();
        return false;  // Never reached, but needed for compiler
    }
    bool addEdgeList(const boost::python::list &edgelist) {
        return addEdgeList_python(edgelist);
    }
    bool addEdgeList_python(const boost::python::list &edgelist) {
        const boost::python::ssize_t elSize = boost::python::len(edgelist);

        // for (auto edge: edgelist) {

        for (boost::python::ssize_t i = 0; i < elSize; ++i) {
            auto edge = edgelist[i];
            addEdge(boost::python::extract<unsigned int>(edge[0]),
                    boost::python::extract<unsigned int>(edge[1])
            );
        }
        return true;
    }
#endif
#ifdef NUMPY
    bool addEdgeListFromNumpyArray(PyObject *npArray) {
        // TODO
        return true;
    }
#endif
    void print() {
        cout << "Graph: " << name << endl;
        for (const auto &[fst, snd]: g) {
            cout << "node " << fst;
            cout << "\t [ ";
            for (const unsigned int &ite: g[fst])
                cout << ite << " ";
            cout << "]" << endl;
        }
    }


    void printCC() {
        cout << "Graph: " << name << endl;
        cout << "Connected Components" << endl;

        // robin_map<unsigned int, vector<unsigned int> *>::iterator itc;
        ComponentType i = 0;

        // TODO: use std::views::enumerate when C++23 is available widely
        // for (auto const &[idx, itc] : std::views::enumerate(component)) {
        // for (const auto &itc: component) {
        for (const auto &[fst, snd]: component) {
            cout << "CC:  " << fst; //itc.first;
            cout << "\t [ ";
            for (const auto &itcc: *snd)//*itc.second)
                cout << itcc << " ";
            cout << "]" << endl;
            ++i;
        }
    }


    void prepareCC() {
        const size_t size = g.size();
        vis.reserve(size);
        stack.reserve(size);
        node2comp.reserve(size);
    }

    void computeCC() {
        cout << "Computing Connected Components..." << endl;
        // robin_map<unsigned int, forward_list<unsigned int> >::iterator itm, mend;
        // forward_list<unsigned int>::iterator itl, lend;
        vector<NodeType> *pCom;

        NodeType numberVisited = 0;
        NodeType snode, node, nnode;

        size_t gSize = g.size();
        cout << "Graph size: " << gSize << endl;

        //initialization
        component.clear();
        counterCC = -1;

        for (const auto &[fst, snd]: g) {
            vis[fst] = false;
        }
        //        for (itm = g.begin(), mend = g.end(); itm != mend; ++itm) {
        //            vis[itm->first] = 0;
        //        }

        for (auto itm = g.begin(), mend = g.end(); (itm != mend) && (numberVisited != gSize); ++itm) {
            snode = itm->first;
            if (!vis[snode]) {
                counterCC++;
                pCom = new vector<NodeType>();
                component[counterCC] = pCom;
                vis[snode] = true;
                stack.push_back(snode);
                while (!stack.empty()) {
                    node = stack.back();
                    stack.pop_back();
                    ++numberVisited;
                    pCom->push_back(node);
                    node2comp[node] = counterCC;
                    for (auto itl = g[node].begin(), lend = g[node].end(); itl != lend; ++itl) {
                        nnode = *itl;
                        if (!vis[nnode]) {
                            vis[nnode] = true;
                            stack.push_back(nnode);
                        }
                    }
                }
            }
        }
    }

    void computeIncCC(const unsigned int idC) {
        NodeType snode, node, nnode;
        NodeType numberVisited = 0;

        vector<NodeType> *pCom = component[idC];
        size_t sizeCC = pCom->size();

        for (auto itv: *pCom) {
            vis[itv] = false;
        }

        for (size_t i = 0; (i < sizeCC) && (numberVisited != sizeCC); ++i) {
            snode = (*component[idC])[i];
            if (!vis[snode]) {
                counterCC++;
                pCom = new vector<NodeType>();
                pCom->reserve(sizeCC);
                component[counterCC] = pCom;
                vis[snode] = true;
                stack.push_back(snode);
                while (!stack.empty()) {
                    node = stack.back();
                    stack.pop_back();
                    ++numberVisited;
                    pCom->push_back(node);
                    node2comp[node] = counterCC;
                    for (const auto& nnode : g[node]) {
                        if (!vis[nnode]) {
                            vis[nnode] = true;
                            stack.push_back(nnode);
                        }
                    }
                }
            }
        }
        delete component[idC];
        component.erase(idC);
    }

    void computeLCCandSLCC(ComponentType &lccID, ComponentType &slccID) {
        // itc;
        // unsigned int max, tmpMax;
        // unsigned int smax;
        // unsigned int maxID, smaxID;

        // cout << "Computing LCC and SLCC..." << endl;
        if (component.empty()) {
            // cout << "No components present." << endl;
            lccID = -1;
            slccID = -1;
            return;
        }

        auto itc = component.begin();

        // Initialize max and maxID with the first component found
        ComponentType maxID = itc->first;
        size_t max = itc->second->size();
        // cout << "Current component ID: " << maxID << " with size " << max << endl;

        size_t smax = 0; // Second largest component size
        ComponentType smaxID = -1; // Second largest component ID
        // cout << "Initializing second largest component size to 0." << endl;

        if (component.size() == 1) {
            // cout << "Only one component present." << endl;
            lccID = maxID;
            slccID = smaxID;
            return;
        }

        for (++itc; itc != component.end(); ++itc) {
            size_t tmpMax = itc->second->size();
            ComponentType tmpID = itc->first;
            // cout << "Current component ID: " << tmpID << " with size " << tmpMax << endl;
            if (tmpMax > max) {
                // If current Component is larger than max, update both max and smax
                smax = max;
                smaxID = maxID;

                max = tmpMax;
                maxID = tmpID;
            } else if ((tmpMax > smax) && (tmpID != maxID)) {
                // If current Component is larger than secondary max but smaller than largest max,
                // update only smax (also check that IDs are different)
                smax = tmpMax;
                smaxID = tmpID;
            }
        }

        lccID = maxID;
        slccID = smaxID;
    }

    vector<NodeType> *getComponent(const ComponentType id) {
        return component[id];
    }

    ComponentType getNodeComp(const NodeType node) {
        return node2comp[node];
    }
};


bool loadNodesFromFile(const string &fname, list<NodeType> &nodes) {
    ifstream inFile;
    int s;

    inFile.open(fname);
    if (!inFile) {
        cout << "Unable to open file";
        return false;
    }

    while (inFile >> s) {
        nodes.push_back(s);
    }
    inFile.close();
    return true;
}


void lccThresholdDismantler(Graph *g, list<NodeType> &nodes, ComponentType stopCondition,
                            vector<tuple<NodeType, NodeType, NodeType> > &removals) {
    vector<NodeType> *pLCC;
    //    vector<int>::iterator result;
    ComponentType lccID, slccID;
    NodeType lccSize = 0, slccSize = 0;

    if (g->isEmpty()) {
        cout << "Graph is empty. Exiting dismantler." << endl;
        return;
    }

    g->prepareCC();
    g->computeCC();
    g->computeLCCandSLCC(lccID, slccID);

    auto it = nodes.begin();
    while (it != nodes.end()) {
        unsigned int nodeToRemove = *it;

        if (g->getNodeComp(nodeToRemove) != lccID) {
            ++it;
            continue;
        }

        // g->clearNode(nodeToRemove);
        g->removeNode(nodeToRemove);
        nodes.erase(it);
        it = nodes.begin();

        //nodeToRemove is still il lccID --> should be removed, but it is costly!!!

        g->computeIncCC(lccID);
        g->computeLCCandSLCC(lccID, slccID);

        pLCC = g->getComponent(lccID);
        lccSize = pLCC->size();
        if (slccID == -1)
            slccSize = 0;
        else
            slccSize = (g->getComponent(slccID))->size();

        //cout << nodeToRemove << " " << lccSize << " " << slccSize << endl;
        removals.emplace_back(nodeToRemove, lccSize, slccSize);

        if (lccSize <= stopCondition)
            break;
    }
}

void thresholdDismantler(Graph *g, list<NodeType> &nodes, NodeType stopCondition,
                         vector<tuple<NodeType, NodeType, NodeType> > &removals) {
    ComponentType lccID, slccID;
    NodeType lccSize = 0, slccSize = 0;

    if (g->isEmpty()) {
        cout << "Graph is empty. Exiting dismantler." << endl;
        return;
    }

    if (g->getFirst()) {
        g->prepareCC();
        g->computeCC();
        g->computeLCCandSLCC(lccID, slccID);
        g->setFirst(false);
    }

    // cout << "Starting dismantling with LCC size: "
    //      << (g->getComponent(lccID))->size() << endl;

    auto it = nodes.begin();
    while (it != nodes.end()) {
        unsigned int nodeToRemove = *it;
        // g->clearNode(nodeToRemove);
        g->removeNode(nodeToRemove);

        //nodeToRemove is still il lccID --> should be removed, but it is costly!!!

        g->computeIncCC(g->getNodeComp(nodeToRemove));
        g->computeLCCandSLCC(lccID, slccID);

        const auto *pLCC = g->getComponent(lccID);
        lccSize = pLCC->size();

        if (slccID == -1)
            slccSize = 0;
        else
            slccSize = (g->getComponent(slccID))->size();

        //cout << nodeToRemove << " " << lccSize << " " << slccSize << endl;
        removals.emplace_back(nodeToRemove, lccSize, slccSize);

        if (lccSize <= stopCondition)
            break;

        ++it;
    }
}


#ifdef BOOST


/*
void unwrapEdgeList(boost::python::list &edgelist, vector<pair<int, int> > &el){
    boost::python::ssize_t elSize = boost::python::len(edgelist);

    for (int i = 0; i < elSize; ++i){
        el.push_back(pair<int, int>(boost::python::extract<int>(edgelist[i][0]), boost::python::extract<int>(edgelist[i][1])));
    }
}
*/

void unwrapList(boost::python::list &nl, list<NodeType> &l) {
    boost::python::ssize_t nlSize = boost::python::len(nl);

    for (int i = 0; i < nlSize; ++i) {
        l.push_back(boost::python::extract<NodeType>(nl[i]));
    }
}

void wrap(vector<tuple<NodeType, NodeType, NodeType> > &removals, boost::python::list &result) {
    for (auto &removal: removals) {
        boost::python::list temp;
        temp.append(get<0>(removal));
        temp.append(get<1>(removal));
        temp.append(get<2>(removal));
        result.append(temp);
    }
}

boost::python::list lcc_dismantler_wrapper(Graph *g, boost::python::list nodesToRemove, ComponentType stopLCC) {
    boost::python::list result;
    vector<tuple<NodeType, NodeType, NodeType> > removals;
    list<NodeType> nodes;

    // cout << "Wrapper LCC Dismantler called with " << boost::python::len(nodesToRemove) << " nodes to remove." << endl;

    if (boost::python::len(nodesToRemove) == 0)
        return result;

    unwrapList(nodesToRemove, nodes);

    lccThresholdDismantler(g, nodes, stopLCC, removals);

    wrap(removals, result);

    return result;
}

boost::python::list dismantler_wrapper(Graph *g, boost::python::list nodesToRemove, const unsigned int stopLCC) {

    boost::python::list result;
    vector<tuple<NodeType, NodeType, NodeType> > removals;
    // vector<pair<unsigned int, unsigned int> > el;
    list<NodeType> nodes;

    // cout << "Wrapper Dismantler called with " << boost::python::len(nodesToRemove) << " nodes to remove." << endl;

    if (boost::python::len(nodesToRemove) == 0)
        return result;

    unwrapList(nodesToRemove, nodes);
    // cout << "Unwrapped " << nodes.size() << " nodes to remove." << endl;

    thresholdDismantler(g, nodes, stopLCC, removals);
    
    // cout << "Dismantling completed with " << removals.size() << " removals." << endl;
    wrap(removals, result);

    // cout << "Wrapped result." << endl;

    return result;
}

#endif


int main(int argc, char **argv) {
    unsigned int deltap = 0;

    if (argc < 5) {
        cout << "Usage: " << argv[0] << " [-lcc] <graph_edgelist> <nodes_to_remove> <output_file> <LCC_size>" << endl;
        return -1;
    }

    if (string(argv[1]) == "-lcc")
        deltap = 1;

    string netFileName(argv[1 + deltap]);
    string nodesFileName(argv[2 + deltap]);
    string outputFileName(argv[3 + deltap]);
    size_t stopLCC = stoi(argv[4 + deltap]);

    list<NodeType> nodes;
    vector<tuple<NodeType, NodeType, NodeType> > removals;
    auto *g = new Graph(netFileName);

#ifdef DEBUG
    cout.setf(std::ios::unitbuf);
    cout << "start loading file...";
#endif

    g->loadEdgeListFromFile(netFileName);

#ifdef DEBUG
    cout << "end!!!" << endl;
#endif

    loadNodesFromFile(nodesFileName, nodes);

#ifdef DEBUG
    clock_t stime, etime;

    cout.setf(std::ios::unitbuf);
    cout << "start dismantling computation" << endl;;
    stime = clock();
#endif

    if (deltap)
        lccThresholdDismantler(g, nodes, stopLCC, removals);
    else
        thresholdDismantler(g, nodes, stopLCC, removals);


#ifdef DEBUG
    etime = clock();
    cout << "end!!!" << endl;
    cout << "time: " << ((float)etime - stime)/CLOCKS_PER_SEC << endl;
#endif

    ofstream outFile(outputFileName);
    for (auto &removal: removals)
        outFile << get<0>(removal) << " " << get<1>(removal) << " " << get<2>(removal) << endl;
    outFile.close();

    delete g;
}


#ifdef BOOST

struct my_exception : std::exception
{
  char const* what() const throw() {
      return "One of my exceptions";
  }
};

void translate(my_exception const& e)
{
    // Use the Python 'C' API to set up an exception object
    PyErr_SetString(PyExc_RuntimeError, e.what());
}

void something_which_throws()
{
    // ...
    throw my_exception();
    // ...
}

BOOST_PYTHON_MODULE (dismantler) {
    using namespace boost::python;
    
    register_exception_translator<my_exception>(&translate);

    auto a = class_<Graph>("Graph", init<>());

    a.def(init<Graph *>())
     .def(init<boost::python::object>())  // Universal constructor: handles string, list, graph_tool.Graph
     
     .def("addNode", &Graph::addNode)
     .def("addNodes", &Graph::addNodes_python)
     .def("addEdge", &Graph::addEdge)
    //  .def("size", &Graph::size)
     .def("addEdgeList", &Graph::addEdgeList_python)
     .def("deepCopy", &Graph::deepCopy, return_value_policy<manage_new_object>())
     .def("lccThresholdDismantler", lcc_dismantler_wrapper)
     .def("thresholdDismantler", dismantler_wrapper)
     .def("print", &Graph::print)
     .def("printCC", &Graph::printCC)
    //  .def("computeCC", &Graph::computeCC);
     .def("getNumNodes", &Graph::getNumNodes)
     .def("getNumEdges", &Graph::getNumEdges)
     .def("isEmpty", &Graph::isEmpty)
     .def("removeNode", &Graph::removeNode)
     .def("clearNode", &Graph::clearNode);

    // Note: Direct GraphInterface and factory function removed to avoid converter conflicts
    // Universal constructor handles all cases (string, list, graph_tool.Graph)

    // Export standalone functions 
    def("lccThresholdDismantler", lcc_dismantler_wrapper);
    def("thresholdDismantler", dismantler_wrapper);

}
#endif
