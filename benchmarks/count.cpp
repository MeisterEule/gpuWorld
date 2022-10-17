#include <stdio.h>
#include <stdlib.h>

#include "compute_step.h"
#include "count.h"
#include "reduction.h"
#include "random.h"

#include "timers.hpp"

#define N_DEFAULT 2048

int main (int argc, char *argv[]) {
	int N = argc > 1 ? atoi(argv[1]) : N_DEFAULT;
	compute_step_t cs_numbers_to_count = new_compute_step (N, N, false, false);;

	Timer tt("CreateNumbers", "ms");
	initRNG (DEFAULT_SEED, N);
	cs_numbers_to_count.data_in = generateRandomArrayInt (N, 0, N-1);
	tt.stop();

	tt.reset("countNumbers");
	countElementsInArray (&cs_numbers_to_count);
        tt.stop();

	//int *count_cpu = countCPU (cs_numbers_to_count.data_in, cs_numbers_to_count.n_data_in);

	//for (int i = 0; i < cs_numbers_to_count.n_data_in; i++) {
	//	//if (cs_numbers_to_count.data_out[i] != count_cpu[i]) printf ("Does not match: %d\n", i);
	//}
	

        compute_step_t cs_count_to_average = cs_from_cs (cs_numbers_to_count, 1, false);

	tt.reset("ComputeSum");
	int sum = computeArraySum (&cs_count_to_average);
	tt.stop();

	printf ("sum: %d\n", sum);

}
