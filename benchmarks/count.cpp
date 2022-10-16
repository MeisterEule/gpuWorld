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
	cs_numbers_to_count.data_in = generateRandomArrayInt (N, 0, N);
	tt.stop();
	//cs_print_in (cs_numbers_to_count);

	tt.reset("countNumbers");
	countElementsInArray (&cs_numbers_to_count);
        tt.stop();
	//cs_print_out (cs_numbers_to_count);

	//compute_step_t cs_count_to_average = cs_from_cs (cs_numbers_to_count, 1, false);
	compute_step_t cs_count_to_average = new_compute_step (N, 1, false, false);
	for (int i = 0; i < N; i++) {
		cs_count_to_average.data_in[i] = 1;
	}

	tt.reset("ComputeSum");
	int sum = computeArraySum (&cs_count_to_average);
	tt.stop();

	printf ("sum: %d\n", sum);

}
