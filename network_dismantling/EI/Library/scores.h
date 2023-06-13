#ifndef STD_H
#define STD_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "macros.h"

#endif

#ifndef NODE_H
#define NODE_H

#include "node.h"

#endif

#ifndef NEWMAN_ZIFF
#define NEWMAN_ZIFF

#include "newman_ziff.h"

#endif


int effective_degree(Node *graph);

int count_number(Node *graph, int id);

double count_sigma2(Node *graph, int id);

double count_sigma1(Node *graph, int id);

int dynam_degree(Node *graph, int id);

extern int *root;
