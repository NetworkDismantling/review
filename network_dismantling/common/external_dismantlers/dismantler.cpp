#include <iostream>
#include <fstream>
#include <forward_list>
#include <list>

#include <Python.h>

// #include "graph_tool.hh"

#ifdef BOOST
#include <boost/python.hpp>

#endif

#include "tsl/robin_map.h"


using namespace std;
using namespace tsl;

class Graph {
private:
    robin_map<unsigned int, forward_list<unsigned int> > g;
    robin_map<unsigned int, bool> vis;
    vector<unsigned int> stack;
    robin_map<unsigned int, vector<unsigned int> *> component;
    //    robin_map<int, vector<unsigned int> *> degrees;
    robin_map<unsigned int, unsigned int> node2comp;
    int counterCC;
    string name;
    bool first = true;

public:
    explicit Graph(string name) {
        this->name = std::move(name);
        counterCC = -1;
    }

#ifdef BOOST
    Graph(boost::python::list pythonList): Graph("") {
        loadGraphFromPythonList(pythonList);
    }
#endif

    explicit Graph(const Graph *graph) : Graph("") {
        g = graph->g;
    }

    ~Graph() {
        // robin_map<int, vector<int> *>::iterator itc, cend;

        for (auto comp: component) {
            delete comp.second;
        }
        //        for (itc = component.begin(), cend = component.end(); itc != cend; ++itc) {
        //            delete itc->second;
        //        }
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

    bool addNode(const unsigned int nodeID) {
        if (g.count(nodeID) > 0) {
            cout << "ERROR :" << nodeID << " already present in the graph: " << name << endl;
            return false;
        } else {
            g[nodeID];
            return true;
        }
    }

    bool removeNode(const unsigned int nodeID) {
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

    bool addEdge(const unsigned int srcNode, const unsigned int dstNode) {
        /*ite = find(g[srcNode].begin(), g[srcNode].end(), dstNode);
        if (ite != g[srcNode].end()){
            cout << "Parallel edge discovered: " << srcNode << " " << dstNode << endl;
            return false;
        }
        */
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

    bool loadEdgeListFromList(const vector<pair<unsigned int, unsigned int> > &el) {
        for (auto &it: el) {
            addEdge(it.first, it.second);
        }
        return true;
    }

#ifdef BOOST
    bool loadGraphFromPythonList(boost::python::list &edgelist) {
        boost::python::ssize_t elSize = boost::python::len(edgelist);

        for (boost::python::ssize_t i = 0; i < elSize; ++i) {
            auto edge = edgelist[i];
            addEdge(boost::python::extract<unsigned int>(edge[0]),
                    boost::python::extract<unsigned int>(edge[1])
            );
        }
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
        cout << "Connectedt Components" << endl;

        robin_map<unsigned int, vector<unsigned int> *>::iterator itc;
        unsigned int i;
        for (itc = component.begin(), i = 0; itc != component.end(); ++itc, ++i) {
            cout << "CC:  " << itc->first;
            cout << "\t [ ";
            for (const unsigned int &itcc: *itc->second)
                cout << itcc << " ";
            cout << "]" << endl;
        }
    }


    void prepareCC() {
        const unsigned long size = g.size();
        vis.reserve(size);
        stack.reserve(size);
        node2comp.reserve(size);
    }

    void computeCC() {
        robin_map<unsigned int, forward_list<unsigned int> >::iterator itm, mend;
        forward_list<unsigned int>::iterator itl, lend;
        vector<unsigned int> *pCom;

        unsigned int numberVisited = 0;
        unsigned int snode, node, nnode;
        unsigned long gSize;

        gSize = g.size();
        //initialization
        component.clear();
        counterCC = -1;

        for (const auto &[fst, snd]: g) {
            vis[fst] = false;
        }
        //        for (itm = g.begin(), mend = g.end(); itm != mend; ++itm) {
        //            vis[itm->first] = 0;
        //        }

        for (itm = g.begin(), mend = g.end(); (itm != mend) && (numberVisited != gSize); ++itm) {
            snode = itm->first;
            if (!vis[snode]) {
                counterCC++;
                pCom = new vector<unsigned int>();
                component[counterCC] = pCom;
                vis[snode] = true;
                stack.push_back(snode);
                while (!stack.empty()) {
                    node = stack.back();
                    stack.pop_back();
                    ++numberVisited;
                    pCom->push_back(node);
                    node2comp[node] = counterCC;
                    for (itl = g[node].begin(), lend = g[node].end(); itl != lend; ++itl) {
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
        forward_list<unsigned int>::iterator itl, lend;

        unsigned int snode, node, nnode;
        unsigned int numberVisited = 0;

        vector<unsigned int> *pCom = component[idC];
        unsigned long sizeCC = pCom->size();

        for (auto itv: *pCom) {
            vis[itv] = false;
        }

        for (unsigned int i = 0; (i < sizeCC) && (numberVisited != sizeCC); ++i) {
            snode = (*component[idC])[i];
            if (!vis[snode]) {
                counterCC++;
                pCom = new vector<unsigned int>();
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
                    for (itl = g[node].begin(), lend = g[node].end(); itl != lend; ++itl) {
                        nnode = *itl;
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

    void computeLCCandSLCC(unsigned int &lccID, unsigned int &slccID) {
        robin_map<unsigned int, vector<unsigned int> *>::iterator itc;
        unsigned int max, tmpMax;
        unsigned int smax;
        unsigned int maxID, smaxID;

        itc = component.begin();
        maxID = itc->first;
        max = itc->second->size();
        smax = 0;
        smaxID = -1;

        for (++itc; itc != component.end(); ++itc) {
            tmpMax = itc->second->size();
            if (tmpMax > max) {
                smax = max;
                smaxID = maxID;

                max = tmpMax;
                maxID = itc->first;
            } else if ((tmpMax > smax) && (tmpMax != max)) {
                smax = tmpMax;
                smaxID = itc->first;
            }
        }

        lccID = maxID;
        slccID = smaxID;
    }

    vector<unsigned int> *getComponent(const unsigned int id) {
        return component[id];
    }

    unsigned int getNodeComp(const unsigned int node) {
        return node2comp[node];
    }
};


bool loadNodesFromFile(const string &fname, list<unsigned int> &nodes) {
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


void lccThresholdDismantler(Graph *g, list<unsigned int> &nodes, unsigned int stopCondition,
                            vector<tuple<unsigned int, unsigned int, unsigned int> > &removals) {
    vector<unsigned int> *pLCC;
    //    vector<int>::iterator result;
    unsigned int lccID, slccID;
    unsigned long lccSize = 0, slccSize = 0;

    g->prepareCC();
    g->computeCC();
    g->computeLCCandSLCC(lccID, slccID);

    list<unsigned int>::iterator it = nodes.begin();
    while (it != nodes.end()) {
        unsigned int nodeToRemove = *it;

        if (g->getNodeComp(nodeToRemove) != lccID) {
            ++it;
            continue;
        }

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

void thresholdDismantler(Graph *g, list<unsigned int> &nodes, unsigned int stopCondition,
                         vector<tuple<unsigned int, unsigned int, unsigned int> > &removals) {
    //    vector<int>::iterator result;
    unsigned int lccID, slccID;
    unsigned int lccSize = 0, slccSize = 0;

    if (g->getFirst()) {
        g->prepareCC();
        g->computeCC();
        g->computeLCCandSLCC(lccID, slccID);
        g->setFirst(false);
    }

    list<unsigned int>::iterator it = nodes.begin();
    while (it != nodes.end()) {
        unsigned int nodeToRemove = *it;
        g->removeNode(nodeToRemove);

        //nodeToRemove is still il lccID --> should be removed, but it is costly!!!

        g->computeIncCC(g->getNodeComp(nodeToRemove));
        g->computeLCCandSLCC(lccID, slccID);

        const vector<unsigned int> *pLCC = g->getComponent(lccID);
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

void unwrapList(boost::python::list &nl, list<unsigned int> &l) {

    boost::python::ssize_t nlSize = boost::python::len(nl);

    for (int i = 0; i < nlSize; ++i) {
        l.push_back(boost::python::extract<unsigned int>(nl[i]));
    }
}

void wrap(vector<tuple<unsigned int, unsigned int, unsigned int> > &removals, boost::python::list &result) {
    for (auto &removal: removals) {
        boost::python::list temp;
        temp.append(get<0>(removal));
        temp.append(get<1>(removal));
        temp.append(get<2>(removal));
        result.append(temp);
    }
}

boost::python::list lcc_dismantler_wrapper(Graph *g, boost::python::list nodesToRemove, unsigned int stopLCC) {
    boost::python::list result;
    vector<tuple<unsigned int, unsigned int, unsigned int> > removals;
    list<unsigned int> nodes;

    unwrapList(nodesToRemove, nodes);

    lccThresholdDismantler(g, nodes, stopLCC, removals);

    wrap(removals, result);

    return result;
}

boost::python::list dismantler_wrapper(Graph *g, boost::python::list nodesToRemove, const unsigned int stopLCC) {

    boost::python::list result;
    vector<tuple<unsigned int, unsigned int, unsigned int> > removals;
    // vector<pair<unsigned int, unsigned int> > el;
    list<unsigned int> nodes;

    unwrapList(nodesToRemove, nodes);

    thresholdDismantler(g, nodes, stopLCC, removals);

    wrap(removals, result);

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
    unsigned int stopLCC = stoi(argv[4 + deltap]);

    list<unsigned int> nodes;
    auto *g = new Graph(netFileName);
    vector<tuple<unsigned int, unsigned int, unsigned int> > removals;

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
BOOST_PYTHON_MODULE (dismantler) {
    using namespace boost::python;

    class_<Graph>("Graph", init<string>())
            .def(init<boost::python::list>())
            .def(init<Graph *>())
            .def("loadGraphFromPythonList", &Graph::loadGraphFromPythonList);

    def("lccThresholdDismantler", lcc_dismantler_wrapper);
    def("thresholdDismantler", dismantler_wrapper);

}
#endif
