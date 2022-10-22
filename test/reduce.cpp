#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#include "compute_step.hpp"
#include "reduction.h"

int doIteration (int N) {
	ComputeStep<int> cs (N, 1, false, false);
        int *data_in = (int*)malloc(N * sizeof(int));
	for (int i = 0; i < N; i++) {
		data_in[i] = 1;
	}
	int *cs_data_in = cs.data_in->front();
	cs_data_in = data_in;
        

	int sum = computeArraySum (&cs);
	//free(cs.data_in);
	//free(cs.data_out);

	return sum;
}

int main (int argc, char *argv[]) {
	int sum;
	for (int N = 1; N < INT_MAX / 2; N *= 2) {
		sum = doIteration (N-1);
		if (sum == N-1) {
			printf ("%d: Okay\n", N);
		} else {
			printf ("%d: Not Okay\n", N);
		}
		sum = doIteration (N);
		if (sum == N) {
			printf ("%d: Okay\n", N);
		} else {
			printf ("%d: Not Okay\n", N);
		}
		sum = doIteration (N+1);
		if (sum == N+1) {
			printf ("%d: Okay\n", N);
		} else {
			printf ("%d: Not Okay\n", N);
		}

	}
	return 0;
}
