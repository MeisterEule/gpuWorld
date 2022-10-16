#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#include "compute_step.h"
#include "reduction.h"

int doIteration (int N) {
	compute_step_t cs = new_compute_step (N, 1, false, false);
	for (int i = 0; i < N; i++) {
		cs.data_in[i] = 1;
	}

	int sum = computeArraySum (&cs);
	free(cs.data_in);
	free(cs.data_out);

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
