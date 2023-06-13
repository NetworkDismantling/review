#ifndef NEWMAN_ZIFF
#define NEWMAN_ZIFF

#include "newman_ziff.h"

#endif

/* Newman-Ziff algorithm for network percolation [ http://dx.doi.org/10.1103/PhysRevE.64.016706 ] */
double newman_ziff(Node *graph, int *largest, double *stilde) {
    int i, j, k, largest_size = 1, nnodes = 0;
    Node *root1, *root2, *aux;
    for (i = 0; i < N; i++) {
        graph[i].pointer = graph + i;
        graph[i].cluster_size = graph[i].n;
        graph[i].visited = (!graph[i].n);
        nnodes += graph[i].n;
    }
    *stilde = nnodes;
    while (true) {
        k = 0;
        i = IRAND(N);
        while (graph[i].visited) {
            i = (i + 1 == N ? 0 : i + 1);
            if (k++ == N) {
                *stilde = *stilde / (1. * N * N);
                return largest_size; //*1.0*/N;//nnodes;
            }
        }
        graph[i].visited = true;
        for (j = 0; j < graph[i].virtual_degree; j++)
            if (graph[i].neigh[j]->n == 1) {
                root1 = findroot(graph + i);
                root2 = findroot(graph[i].neigh[j]);
                if (root1->id != root2->id) {
                    if (root1->cluster_size > root2->cluster_size) {
                        aux = root1;
                        root1 = root2;
                        root2 = aux;
                    }
                    root1->pointer = root2;
                    *stilde += 2 * root2->cluster_size * root1->cluster_size;
                    root2->cluster_size += root1->cluster_size;
                    if (largest_size < root2->cluster_size) {
                        largest_size = root2->cluster_size;
                        *largest = root2->id;
                    }
                }
            }
    }
}

double newman_ziff_adapted(Node *graph) {
    int i, j, k, largest_size = 1, nnodes = 0;
    Node *root1, *root2, *aux;

    for (i = 0; i < N; i++) {
        graph[i].pointer = graph + i;
        graph[i].cluster_size = 1;
        graph[i].visited = (!graph[i].n);
        nnodes += graph[i].n;
    }

    while (true) {
        k = 0;
        i = IRAND(N);
        while (graph[i].visited) {
            i = (i + 1 == N ? 0 : i + 1);
            if (k++ == N) {
                return largest_size; //1.0*/N;
            }
        }
        graph[i].visited = true;
        for (j = 0; j < graph[i].virtual_degree; j++)
            if (graph[i].neigh[j]->n == 1) {
                root1 = findroot(graph + i);
                root2 = findroot(graph[i].neigh[j]);
                if (root1->id != root2->id) {
                    if (root1->cluster_size > root2->cluster_size) {
                        aux = root1;
                        root1 = root2;
                        root2 = aux;
                    }
                    root1->pointer = root2;
                    root2->cluster_size += root1->cluster_size;
                    if (largest_size < root2->cluster_size) {
                        largest_size = root2->cluster_size;
                    }
                }
            }
    }
}

double newman_ziff_back(Node *graph, int *largest, Node *added, double *stilde) {
    int i, aux2;
    Node *root;
    unsigned long aux = 1;
    unsigned long a1, a2;

    added->cluster_size = 1;
    added->pointer = added;
    for (i = 0; i < added->virtual_degree; i++)
        if (added->neigh[i]->n == 1) {
            root = findroot(added->neigh[i]);
            if (root->id != added->id) {
                root->pointer = added;
                a1 = added->cluster_size;
                a2 = root->cluster_size;
                aux = aux + 2 * a1 * a2;
                added->cluster_size += root->cluster_size;
            }
        }

    if (added->cluster_size >= graph[*largest].cluster_size) {
        *largest = added->id;
    }

    if (aux < 0)fprintf(stderr, "\nnewman_ziff_back (Warning!): integer overflow!\n\n");
    *stilde += 1. * aux / (1.0 * N * N);

    return graph[*largest].cluster_size; //*1.0/(1.*N);
}

Node *findroot(Node *node) {
    if (node->pointer->id != node->id) {
        node->pointer = findroot(node->pointer);
        return node->pointer;
    }
    return node;
}
