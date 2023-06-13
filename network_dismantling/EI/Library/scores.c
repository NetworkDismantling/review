#include "scores.h"

int effective_degree(Node *graph) {
    int i, j, *effective, flag = 0;
    int count = 0;
    effective = (int *) malloc(sizeof(int) * N);

    for (i = 0; i < N; i++)
        effective[i] = graph[i].virtual_degree;

    while (flag == 0) {
        count++;
        flag = 1;
        for (i = 0; i < N; i++) {
            graph[i].effective_degree = 0;
            for (j = 0; j < graph[i].virtual_degree; j++)
                if (graph[i].neigh[j]->effective_degree <= eff_thr) // if neighbour j is not a hub
                    if (graph[i].neigh[j]->virtual_degree > 1) // if neighbour j is not a leave
                        graph[i].effective_degree++; // increase node's i effective degree
        }

        for (i = 0; i < N; i++) {
            if (graph[i].effective_degree != effective[i])
                flag = 0;
            effective[i] = graph[i].effective_degree; //we keep doing this until the effective degree does not change for any node.
        }
    }
    fprintf(stderr, "Effective degree loops: %d\n", count);
    free(effective);
    return 1;
}

int count_number(Node *graph, int id) {
    int c1 = 0, c2 = 0, j, i, k = 0, largest_size = 0;

    for (j = 0; j < graph[id].virtual_degree; j++) //for each neighbour
        if (graph[id].neigh[j]->n == 1) { //that is already in the network (unvaccinated)
            root[k] = findroot(graph[id].neigh[j])->id; //take its root
            for (i = 0; i < k; i++) // and see if the corresponding cluster
                if (root[i] == root[k]) break; // has been taken already
            k += (i == k ? 1 : 0); //if not increase k only if the cluster is new
        }

    return c2;
}


double count_sigma2(Node *graph, int id) {
    int c1 = 0, c2 = 0, j, i, k = 0, largest_size = 0, aux;

    for (j = 0; j < graph[id].virtual_degree; j++) //for each neighbour
        if (graph[id].neigh[j]->n == 1) { //that is already in the network (unvaccinated)
            root[k] = findroot(graph[id].neigh[j])->id; //take its root
            for (i = 0; i < k; i++) // and see if the corresponding cluster
                if (root[i] == root[k]) break; // has been taken already
            if (i == k) { //if not,
                aux = root[k];
                if (graph[aux].cluster_size > c1) { // see if its the largest cluster
                    c2 = c1;
                    c1 = graph[aux].cluster_size;
                } else if (graph[aux].cluster_size > c2) //or the second largest
                    c2 = graph[aux].cluster_size;
                k++; //increase k only if the cluster is new
            }
        }

    return k + 1. * c2 / N;
}

double count_sigma1(Node *graph, int id) {
    int j, i, k = 0;
    double c = 0;

    for (j = 0; j < graph[id].virtual_degree; j++) //for each neighbour
        if (graph[id].neigh[j]->n == 1) { //that is already in the network (unvaccinated)
            root[k] = findroot(graph[id].neigh[j])->id; //take its root
            for (i = 0; i < k; i++) //if the root has not taken already
                if (root[i] == root[k]) break;
            if (k == i) {
                c += sqrt(1. * graph[root[k]].cluster_size) - 1; // perform the counting
                k++; //increase k only if the cluster is new
            }
        }

    return graph[id].effective_degree + c;
}

