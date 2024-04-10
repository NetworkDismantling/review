#   This file is part of CoreGDM (Core Graph Dismantling with Machine learning),
#   proposed in the paper "CoreGDM: Geometric Deep Learning Network Decycling
#   and Dismantling" by M. Grassia and G. Mangioni.
#
#   CoreGDM is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   CoreGDM is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with CoreGDM.  If not, see <http://www.gnu.org/licenses/>.

# The original tree_breaker function is taken from the following repository:
# https://github.com/abraunst/decycler

# quick and dirty graph implementation: removed nodes are just flagged

import heapq


class Graph:
    V: list = []
    present: list = []
    M: int = 0

    def __init__(self):
        self.V = []
        self.present = []
        self.M = 0

    def size(self):
        return self.present.count(True)

    def add_node(self, i):
        if i >= len(self.V):
            delta = i + 1 - len(self.V)
            self.present += [False] * delta
            # self.present += [True] * delta
            self.V += [[] for _ in range(delta)]

        self.present[i] = True

    def add_edge(self, i, j):
        self.add_node(i)
        self.add_node(j)

        self.V[i] += [j]
        self.V[j] += [i]

        self.M += 1

    def remove_node(self, i):
        self.present[i] = False
        self.M -= sum(1 for j in self.V[i] if (self.present[j] is True))


class CyclesError(RuntimeError):
    pass


def tree_breaker(G: Graph, stop_condition: int):
    N = G.size()
    S = [0] * len(G.V)

    def size(i, j):
        if G.present[i] is False:
            return 0

        if S[i] != 0:
            raise CyclesError("the graph is NOT acyclic", i, S[i], j)

        S[i] = 1 + sum(
            size(k, i) for k in G.V[i] if ((k != j) and (G.present[k] is True))
        )

        return S[i]

    H = [
        (-size(i, None), i)
        for i in range(len(G.V))
        if (G.present[i] is True) and (not S[i])
    ]

    Ncc = len(H)
    # print("# N:", N, "Ncc:", Ncc, "M:", G.M)
    assert N - Ncc == G.M

    heapq.heapify(H)

    output = []
    while len(H):
        s, i = heapq.heappop(H)
        scomp = -s
        sender = None

        while True:
            sizes = [
                (S[k], k) for k in G.V[i] if (k != sender) and (G.present[k] is True)
            ]

            if len(sizes) == 0:
                break

            M, largest = max(sizes)

            if M <= scomp / 2:
                for k in G.V[i]:
                    if (S[k] > 1) and (G.present[k] is True):
                        heapq.heappush(H, (-S[k], k))

                # n += 1
                # print("S {} {} {}".format(i, n, scomp))

                G.remove_node(i)
                output.append(i)

                if scomp <= stop_condition:
                    return output

                break

            S[i] = 1 + sum(
                S[k] for k in G.V[i] if (k != largest) and (G.present[k] is True)
            )

            sender, i = i, largest

    return output
