#ifndef STD_H
#define STD_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <errno.h>
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

#ifndef NETWORKS_H
#define NETWORKS_H

#include "networks.h"

#endif

#ifndef SCORES_H
#define SCORES_H

#include "scores.h"

#endif

#ifndef NEWMAN_ZIFF_H
#define NEWMAN_ZIFF_H

#include "newman_ziff.h"

#endif

int read_network(char *namefile);

int explosive_immunization(int threshold, int sigma, int nn, char *output_file, char *threshold_condition_file);

int make_network();

int print_threshold_conditions();
