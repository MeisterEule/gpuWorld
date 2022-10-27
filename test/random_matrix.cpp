#include <stdio.h>

#include "memoryManager.hpp"
#include "random.h"
#include "count.h"

int main (int argc, char *argv[]) {

	int N = argc > 1 ? atoi(argv[1]) : 1000;

	memoryManager *mm = new memoryManager(false);
	cudaRNG *rng = new cudaRNG (1024 * 1024, DEFAULT_SEED);

	rng->initRNG (mm, N);
	printf ("stride: %d\n", rng->gen_stride);
	printf ("tmax: %d\n", rng->n_threads * (rng->n_blocks - 1) + (rng->n_threads - 1) * (rng->gen_stride + 1));

	float nonzero_ratio = 0.5;
	float *numbers = rng->generateRandomMatrix (mm, N, nonzero_ratio);

	int nnz = countNonzeros (mm, numbers, N, false);
	
	printf ("Desired ratio: %f\n", nonzero_ratio);
	printf ("Generated ratio: %f\n", (float)nnz / N);
	printf ("Count on host...\n");
	nnz = 0;
	for (int i = 0; i < N; i++) {
		if (numbers[i] > 0) nnz++;
		//printf ("%f\n", numbers[i]);
	}
	printf ("nnz: %f\n", (float)nnz / N);
}
