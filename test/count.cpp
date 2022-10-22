#include <stdio.h>
#include <stdlib.h>

#include "compute_step.hpp"
#include "memoryManager.hpp"
#include "count.h"

int main (int argc, char *argv[]) {
	if (argc > 1) {
		memoryManager *mm = new memoryManager();

		int N = atoi(argv[1]);
		int *data_in = (int*)malloc(N * sizeof(int));
		for (int i = 0; i < N; i++) { 
			data_in[i] = i;
		}
	        ComputeStep<int> cs1 (mm, N, N, false, false, data_in);

		
		countElementsInArray (mm, &cs1);

		int n_fail = 0;
		int *data_out = cs1.data_out->front();
		for (int i = 0; i < N; i++) {
			if (data_out[i] != 1) {
				n_fail++;
			}
		}
		if (n_fail == 0) {
		   printf ("OK\n");
		} else {
		   printf ("Fail: %d\n", n_fail);
		}
		return 0;
	}
	return -1;
}
