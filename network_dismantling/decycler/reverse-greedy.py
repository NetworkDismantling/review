import networkx as nx
import sys

G = nx.Graph()
S = set()

n = 0
for l in sys.stdin:
    v = l.split()
    if v[0] == "D":
        G.add_edge(int(v[1]), int(v[2]))
    if v[0] == "S":
        S.add(int(v[1]))
        n += 1

N = len(G)
while True:
    cc = list(nx.connected_component_subgraphs(G))
    largest = len(cc[0])
    comp = {}
    for c in cc:
        for i in c:
            comp[i] = c
    bestnum = len(G)
    s = 1
    mins = N
    for i in S:
        comps = set()
        for j in G[i]:
            if comp[j] not in comps:
                s += len(comp[j])
                comps.add(comp[j])
        if s < mins:
            mins = s
            mini = i
    S.remove(mini)
    print(mini, largest)
    sys.stdout.flush()
