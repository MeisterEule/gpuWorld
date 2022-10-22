#include <stdio.h>
#include <stdlib.h>

#include "compute_step.hpp"
#include "count.h"
#include "reduction.h"
#include "random.h"

#include "timers.hpp"

#define N_DEFAULT 2048

int main (int argc, char *argv[]) {
	int N = argc > 1 ? atoi(argv[1]) : N_DEFAULT;
	Timer tt("CreateNumbers", "ms");
	initRNG (DEFAULT_SEED, N);
	ComputeStep<int> cs_numbers_to_count (N, N, false, false, generateRandomArrayInt (N, 0, N-1));

	//cs_numbers_to_count.SetInFirst(generateRandomArrayInt (N, 0, N-1));
	tt.stop();

	tt.reset("countNumbers");
	countElementsInArray (&cs_numbers_to_count);
        tt.stop();

        ComputeStep<int> cs_count_to_average (cs_numbers_to_count, 1, false);

	tt.reset("ComputeSum");
	int sum = computeArraySum (&cs_count_to_average);
	tt.stop();

	printf ("sum: %d\n", sum);

}
