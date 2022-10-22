#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#include "memoryManager.hpp"
#include "compute_step.hpp"
#include "reduction.h"

memoryManager *mm;

int doIteration (int N) {
	int *data_in = (int*)malloc(N * sizeof(int));
	for (int i = 0; i < N; i++) {
		data_in[i] = 1;
	}
	ComputeStep<int> cs (mm, N, 1, false, false, data_in);

	int sum = computeArraySum (mm, &cs);

	return sum;
	return 0;
}

int main (int argc, char *argv[]) {
	int sum;
	mm = new memoryManager(false);
	int N_max = argc > 1 ? atoi(argv[1]) : INT_MAX / 2;
	for (int N = 1; N <= N_max; N *= 2) {
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
