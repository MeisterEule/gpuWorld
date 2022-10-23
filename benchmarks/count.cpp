#include <stdio.h>
#include <stdlib.h>

#include "compute_step.hpp"
#include "timers.hpp"
#include "memoryManager.hpp"
#include "arrayGenerators.hpp"

#include "count.h"
#include "reduction.h"

#define N_DEFAULT 2048

int main (int argc, char *argv[]) {
	int N = argc > 1 ? atoi(argv[1]) : N_DEFAULT;
	memoryManager *mm = new memoryManager(true);
	Timer tt("CreateNumbers", "ms");
	cudaRNG *rng = new cudaRNG (1024 * 1024, DEFAULT_SEED);
	rng->initRNG (mm, N);
	ComputeStep<int,int> cs_numbers_to_count (mm, N, N, false, false, rng->generate (mm, N, 0, N-1));

	tt.stop();

	tt.reset("countNumbers");
	countElementsInArray (mm, &cs_numbers_to_count);
        tt.stop();

        ComputeStep<int,int> cs_count_to_average (mm, cs_numbers_to_count, 1, false);

	tt.reset("ComputeSum");
	int sum = computeArraySum (mm, &cs_count_to_average);
	tt.stop();

	printf ("sum: %d\n", sum);

}
