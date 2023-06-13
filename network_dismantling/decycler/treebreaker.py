import sys
import heapq


# quick and dirty graph implementation: removed nodes are just flagged


class Graph:
    V = []
    present = []
    M = 0

    def size(self):
        return sum(self.present)

    def add_node(self, i):
        if i >= len(self.V):
            delta = i + 1 - len(self.V)
            self.present += [1] * delta
            self.V += [[] for j in range(delta)]

    def add_edge(self, i, j):
        self.add_node(i)
        self.add_node(j)
        self.V[i] += [j]
        self.V[j] += [i]
        self.M += 1

    def remove_node(self, i):
        self.present[i] = 0
        self.M -= sum(1 for j in self.V[i] if self.present[j])


G = Graph()

n = 0
for l in sys.stdin:
    v = l.split()
    if v[0] == 'D' or v[0] == 'E':
        G.add_edge(int(v[1]), int(v[2]))
    if v[0] == "V":
        G.add_node(int(v[1]))
    if v[0] == 'S':
        G.remove_node(int(v[1]))
        n += 1

N = G.size()
S = [0] * len(G.V)


def size(i, j):
    if not G.present[i]:
        return 0
    if S[i] != 0:
        # print("# the graph is NOT acyclic")
        # exit()
        print(i, S[i], j)
        exit("the graph is NOT acyclic")
    S[i] = 1 + sum(size(k, i) for k in G.V[i] if (k != j and G.present[k]))
    return S[i]


H = [(-size(i, None), i) for i in range(len(G.V)) if G.present[i] and not S[i]]

Ncc = len(H)
# print("# N:", N, "Ncc:", Ncc, "M:", G.M)
assert (N - Ncc == G.M)

# print("# the graph is acyclic")

sys.stdout.flush()

heapq.heapify(H)
while len(H):
    s, i = heapq.heappop(H)
    scomp = -s
    sender = None
    while True:
        sizes = [(S[k], k) for k in G.V[i] if k != sender and G.present[k]]
        if len(sizes) == 0:
            break
        M, largest = max(sizes)
        if M <= scomp / 2:
            for k in G.V[i]:
                if S[k] > 1 and G.present[k]:
                    heapq.heappush(H, (-S[k], k))
            G.remove_node(i)
            n += 1

            print("S {} {} {}".format(i, n, scomp))
            if scomp <= int(sys.argv[1]):
                exit()
            sys.stdout.flush()
            break
        S[i] = 1 + sum(S[k] for k in G.V[i] if k != largest and G.present[k])
        sender, i = i, largest
