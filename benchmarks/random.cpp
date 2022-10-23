#include <stdlib.h>

#include "memoryManager.hpp"

#include "count.h"
#include "random.h"

int main (int argc, char *argv[]) {
	int N = argc > 1 ? atoi(argv[1]) : 10000;
	memoryManager *mm = new memoryManager(true);
	cudaRNG *rng = new cudaRNG(1024 * 1024, DEFAULT_SEED);
	rng->initRNG (mm, N);
	int *random_numbers = rng->generate (mm, N, 0, 100);

	free(random_numbers);	
}
