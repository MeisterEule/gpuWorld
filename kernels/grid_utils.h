#ifndef GRID_UTILS_H
#define GRID_UTILS_H

#include "types.h"

#define GRID_MAX_THREADS 1024

void getGridDimension1D (LDIM N_total, int *n_blocks, int *n_threads);

#endif
