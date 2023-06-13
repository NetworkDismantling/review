#ifndef STD_H
#define STD_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
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

#ifndef NEWMAN_ZIFF
#define NEWMAN_ZIFF

#include "newman_ziff.h"

#endif

Node *read_net(char *name);

int reset_net(Node *graph, char *threshold_condition_file);
