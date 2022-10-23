#include <stdio.h>

#include "memoryManager.hpp"
#include "random.h"

int main (int argc, char *argv[]) {
	int N = argc > 1 ? atoi(argv[1]) : 1000;

	memoryManager *mm = new memoryManager(false);
	cudaRNG *rng = new cudaRNG (1024 * 1024, DEFAULT_SEED);

	rng->initRNG (mm, N);
	int *numbers = rng->generate (mm, N, 0, N-1);

	for (int i = 0; i < N; i++) {
		if (!(numbers[i] >= 0 && numbers[i] < N)) printf ("Not okay: %d %d\n", i, numbers[i]);
	}
	return 0;
}
