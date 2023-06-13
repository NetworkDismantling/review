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

#define RAND (((double)rand())/RAND_MAX)    // random double beween 0 and 1
#define IRAND(n) ((int) n*RAND)        // random double between -1 and 1
#define RANDIFF (-1+2.*RAND)            // random integer between 0 and n-1 (inclusive)

void initialize_rng(int seed);
