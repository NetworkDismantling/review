#ifndef STD_H
#define STD_H

#include<stdlib.h>
#include <stdio.h>
#include "macros.h"

#endif

#ifndef RNG_H
#define RNG_H

#include "rng.h"

#endif

#ifndef NODE_H
#define NODE_H

#include "node.h"

#endif

double newman_ziff_back(Node *graph, int *largest, Node *added, double *stilde2);

double newman_ziff(Node *graph, int *largest, double *stilde2);

Node *findroot(Node *node);

double newman_ziff_adapted(Node *graph);
