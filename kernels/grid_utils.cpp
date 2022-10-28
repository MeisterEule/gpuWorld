#include "types.h"
#include "grid_utils.h"

void getGridDimension1D (LDIM N_total, int *n_blocks, int *n_threads) {
	if (N_total > GRID_MAX_THREADS) {
                *n_blocks = (N_total + GRID_MAX_THREADS - 1) / GRID_MAX_THREADS;
                *n_threads = GRID_MAX_THREADS;
        } else {
                *n_threads = N_total;
                *n_blocks = 1;
        }
}
