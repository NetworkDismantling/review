#include <iostream>
#include <algorithm>
#include <fstream>
//#include <string>
#include <unordered_map>
#include <forward_list>
#include <list>
#include <vector>
//#include <tuple>
//#include <chrono>
#include "tsl/robin_map.h"


#ifdef BOOST
#include <boost/python.hpp>
#endif

using namespace std;
using namespace tsl;

class Graph {
private:
    robin_map<int, forward_list<int> > g;
    robin_map<int, int> vis;
    vector<int> stack;
    robin_map<int, vector<int> *> comp;
    robin_map<int, int> node2comp;
    int counterCC;
    string name;
    bool first = true;

public:
    explicit Graph(string name) {
        this->name = name;
        counterCC = -1;
    }

#ifdef BOOST
    Graph(boost::python::list pythonList): Graph(""){
        loadGraphFromPythonList(pythonList);
    }
#endif

    explicit Graph(Graph *graph) : Graph("") {
        g = graph->g;
    }

    ~Graph() {
        robin_map<int, vector<int> *>::iterator itc, cend;

        for (itc = comp.begin(), cend = comp.end(); itc != cend; ++itc) {
            delete itc->second;
        }
    }

    bool getFirst() {
        return first;
    }

    void setFirst(bool val) {
        first = val;
    }

    int size() {
        return g.size();
    }

    bool addNode(int nodeID) {
        if (g.count(nodeID) > 0) {
            cout << "ERROR :" << nodeID << " already present in the graph: " << name << endl;
            return false;
        } else {
            g[nodeID];
            return true;
        }
    }

    bool removeNode(int nodeID) {
        forward_list<int>::iterator ite;

        if (g.count(nodeID) == 0) {
            cout << "ERROR: Node " << nodeID << " not present in the graph: " << name << endl;
            return false;
        } else {
            for (ite = g[nodeID].begin(); ite != g[nodeID].end(); ++ite)
                g[*ite].remove(nodeID);

            g.erase(nodeID);
            return true;
        }
    }

    bool addEdge(int srcNode, int dstNode) {
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

    bool removeEdge(int srcNode, int dstNode) {
        g[srcNode].remove(dstNode);
        g[dstNode].remove(srcNode);
        return true;
    }

    bool loadEdgeListFromFile(string &fname) {
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

    bool loadEdgeListFromList(vector<pair<int, int>> el) {

        for (vector<pair<int, int> >::iterator it = el.begin(); it != el.end();
             ++it) {
            addEdge(it->first, it->second);
        }
        return true;
    }

#ifdef BOOST
    bool loadGraphFromPythonList(boost::python::list &edgelist){
        boost::python::ssize_t elSize = boost::python::len(edgelist);

        for (int i = 0; i < elSize; ++i){
            addEdge(boost::python::extract<int>(edgelist[i][0]), boost::python::extract<int>(edgelist[i][1]));
    }
        return true;
    }
#endif

    void print() {
        robin_map<int, forward_list<int> >::iterator
                itn;
        forward_list<int>::iterator ite;


        cout << "Graph: " << name << endl;
        for (itn = g.begin(); itn != g.end(); ++itn) {
            cout << "node " << itn->first;
            cout << "\t [ ";
            for (ite = g[itn->first].begin(); ite != g[itn->first].end(); ++ite)
                cout << *ite << " ";
            cout << "]" << endl;
        }
    }


    void printCC() {
        robin_map<int, vector<int> *>::iterator itc;
        vector<int>::iterator itcc;
        int i;

        cout << "Graph: " << name << endl;
        cout << "Connectedt Components" << endl;
        for (itc = comp.begin(), i = 0; itc != comp.end(); ++itc, ++i) {
            cout << "CC:  " << itc->first;
            cout << "\t [ ";
            for (itcc = itc->second->begin(); itcc != itc->second->end(); ++itcc)
                cout << *itcc << " ";
            cout << "]" << endl;
        }
    }


    void prepareCC() {
        unsigned long size = g.size();
        vis.reserve(size);
        stack.reserve(size);
        node2comp.reserve(size);
    }

    void computeCC() {
        robin_map<int, forward_list<int> >::iterator itm, mend;
        forward_list<int>::iterator itl, lend;
        vector<int> *pCom;

        int numberVisited = 0;
        int snode, node, nnode, gSize;

        gSize = g.size();
        //initialization
        comp.clear();
        counterCC = -1;

        for (itm = g.begin(), mend = g.end(); itm != mend; ++itm) {
            vis[itm->first] = 0;
        }

        for (itm = g.begin(), mend = g.end(); (itm != mend) && (numberVisited != gSize); ++itm) {
            snode = itm->first;
            if (!vis[snode]) {
                counterCC++;
                pCom = new vector<int>();
                comp[counterCC] = pCom;
                vis[snode] = 1;
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
                            vis[nnode] = 1;
                            stack.push_back(nnode);
                        }
                    }
                }
            }
        }
    }


    void computeIncCC(int idC) {
        forward_list<int>::iterator itl, lend;
        vector<int>::iterator itv, vend;
        vector<int> *pCom;

        int snode, node, nnode;
        unsigned long sizeCC;
        int i;
        int numberVisited = 0;


        pCom = comp[idC];
        sizeCC = pCom->size();

        for (itv = pCom->begin(), vend = pCom->end(); itv != vend; ++itv) {
            vis[*itv] = 0;
        }

        for (i = 0; (i < sizeCC) && (numberVisited != sizeCC); ++i) {
            snode = (*comp[idC])[i];
            if (!vis[snode]) {
                counterCC++;
                pCom = new vector<int>();
                pCom->reserve(sizeCC);
                comp[counterCC] = pCom;
                vis[snode] = 1;
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
                            vis[nnode] = 1;
                            stack.push_back(nnode);
                        }
                    }
                }
            }
        }
        delete comp[idC];
        comp.erase(idC);
    }

    void computeLCCandSLCC(int &lccID, int &slccID) {
        robin_map<int, vector<int> *>::iterator itc;
        unsigned long max, tmpMax;
        unsigned long smax;
        int maxID, smaxID;

        itc = comp.begin();
        maxID = itc->first;
        max = itc->second->size();
        smax = 0;
        smaxID = -1;

        itc++;
        for (; itc != comp.end(); ++itc) {
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

    vector<int> *getComponent(int id) {
        return comp[id];
    }

    int getNodeComp(int node) {
        return node2comp[node];
    }

};


bool loadNodesFromFile(string fname, list<int> &nodes) {
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


void lccThresholdDismantler(Graph *g, list<int> &nodes, int stopCondition, vector<tuple<int, int, int>> &removals) {
    int nodeToRemove;
    vector<int> *pLCC;
//    vector<int>::iterator result;
    list<int>::iterator it;
    int lccID, slccID;
    unsigned long lccSize = 0, slccSize = 0;

    g->prepareCC();
    g->computeCC();
    g->computeLCCandSLCC(lccID, slccID);

    it = nodes.begin();
    while (it != nodes.end()) {
        nodeToRemove = *it;

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
        removals.push_back(tuple<int, int, int>(nodeToRemove, lccSize, slccSize));

        if (lccSize <= stopCondition)
            break;

    }

}

void thresholdDismantler(Graph *g, list<int> &nodes, int stopCondition, vector<tuple<int, int, int>> &removals) {
    int nodeToRemove;
    vector<int> *pLCC;
//    vector<int>::iterator result;
    list<int>::iterator it;
    int lccID, slccID;
    unsigned long lccSize = 0, slccSize = 0;

    if (g->getFirst()) {
        g->prepareCC();
        g->computeCC();
        g->computeLCCandSLCC(lccID, slccID);
        g->setFirst(false);
    }

    it = nodes.begin();
    while (it != nodes.end()) {
        nodeToRemove = *it;
        g->removeNode(nodeToRemove);

        //nodeToRemove is still il lccID --> should be removed, but it is costly!!!

        g->computeIncCC(g->getNodeComp(nodeToRemove));
        g->computeLCCandSLCC(lccID, slccID);

        pLCC = g->getComponent(lccID);
        lccSize = pLCC->size();

        if (slccID == -1)
            slccSize = 0;
        else
            slccSize = (g->getComponent(slccID))->size();

        //cout << nodeToRemove << " " << lccSize << " " << slccSize << endl;
        removals.push_back(tuple<int, int, int>(nodeToRemove, lccSize, slccSize));

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
}*/

void unwrapList(boost::python::list &nl, list<int> &l){
    boost::python::ssize_t nlSize = boost::python::len(nl);

    for (int i = 0; i < nlSize; ++i){
        l.push_back(boost::python::extract<int>(nl[i]));
    }
}

void wrap(vector<tuple<int, int, int> > &removals, boost::python::list &result){

    for (vector<tuple<int, int, int> >::iterator it = removals.begin(); it != removals.end(); ++it){
        boost::python::list temp;
        temp.append(get<0>(*it));
        temp.append(get<1>(*it));
        temp.append(get<2>(*it));
        result.append(temp);
    }
}

boost::python::list lcc_dismantler_wrapper(Graph *g, boost::python::list nodesToRemove, int stopLCC){

    boost::python::list result;
    vector<tuple<int, int, int> > removals;
    list<int> nodes;


    unwrapList(nodesToRemove, nodes);

    lccThresholdDismantler(g, nodes, stopLCC, removals);

    wrap(removals, result);
    return result;

}

boost::python::list dismantler_wrapper(Graph *g, boost::python::list nodesToRemove, int stopLCC){

    boost::python::list result;
    vector<tuple<int, int, int> > removals;
    vector<pair<int, int> >el;
    list<int> nodes;


    unwrapList(nodesToRemove, nodes);

    thresholdDismantler(g, nodes, stopLCC, removals);

    wrap(removals, result);
    return result;

}

#endif



int main(int argc, char **argv) {
    int deltap = 0;

    if (argc < 5) {
        cout << "Usage: " << argv[0] << " [-lcc] <graph_edgelist> <nodes_to_remove> <output_file> <LCC_size>" << endl;
        return -1;
    }

    if (string(argv[1]).compare("-lcc") == 0)
        deltap = 1;

    string netFileName(argv[1 + deltap]);
    string nodesFileName(argv[2 + deltap]);
    string outputFileName(argv[3 + deltap]);
    int stopLCC = stoi(argv[4 + deltap]);

    list<int> nodes;
    Graph *g = new Graph(netFileName);
    vector<tuple<int, int, int>> removals;

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
    for (vector<tuple<int, int, int> >::iterator it = removals.begin(); it != removals.end(); ++it)
        outFile << get<0>(*it) << " " << get<1>(*it) << " " << get<2>(*it) << endl;
    outFile.close();

}


#ifdef BOOST
BOOST_PYTHON_MODULE(dismantler)
{
    using namespace boost::python;
        class_<Graph>("Graph", init<string>())
            .def(init<boost::python::list>())
            .def(init<Graph*>())
        .def("loadGraphFromPythonList", &Graph::loadGraphFromPythonList)
        ;
    def("lccThresholdDismantler", lcc_dismantler_wrapper);
    def("thresholdDismantler", dismantler_wrapper);
}
#endif
