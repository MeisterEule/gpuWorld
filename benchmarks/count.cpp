#include <stdio.h>
#include <stdlib.h>

#include "compute_step.h"
#include "count.h"
#include "reduction.h"

#include "timers.hpp"

int main (int argc, char *argv[]) {
	int N = 2048;
	compute_step_t cs_numbers_to_count = new_compute_step (N, N, false, false);;

	for (int i = 0; i < N; i++) {
		cs_numbers_to_count.data_in[i] = i % 3;
	}
    	

	Timer tt("CountElements", "ms");
	countElementsInArray (&cs_numbers_to_count);
        tt.stop();

	//int nnz = 0;
   	//for (int i = 0; i < N; i++) {
	//	printf ("%d: %d\n", cs_numbers_to_count.data_in[i], cs_numbers_to_count.data_out[i]);
	//	if (cs_numbers_to_count.data_out[i] > 0) nnz++;
	//}

	compute_step_t cs_count_to_average = cs_from_cs (cs_numbers_to_count, 1, false);

	//computeAverageOfArray (&cs_count_to_average);
	tt.reset("ComputeSum");
	int sum = computeArraySum (&cs_count_to_average);
	tt.stop();

	printf ("SUM: %d\n", sum);
	//printf ("Sum: %d, Average: %f\n", cs_count_to_average.data_out[0], (float)(cs_count_to_average.data_out[0]) / N);

	//cs_free (&cs_numbers_to_count);
	//cs_free (&cs_count_to_average);

}
