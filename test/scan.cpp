#include "memoryManager.hpp"
#include "compute_step.hpp"
#include "arrayGenerators.hpp"

#include "scan.h"
#include "reduction.h"

int main (int argc, char *argv[]) {
	int N = argc > 1 ? atoi(argv[1]) : 1000;

	if (N > 1024) {
		printf ("Scan for more than one block are not implemented yet. Choose N <= 1024.\n");
		return -1;
	}

	memoryManager *mm = new memoryManager(false);
	ComputeStep<int,int> cs (mm, N, N, false, false, generateArrayOfOnesCPU<int> (N));

	scanArray (mm, &cs, false);

	int *data_out = cs.GetOutFirst();

	int n_not_okay = 0; 
	for (int i = 0; i < N; i++) {
		if (data_out[i] != i + 1) {
			printf ("Not okay: %d %d\n", i, data_out[i]);
			n_not_okay++;
		}
	}
	if (n_not_okay == 0) printf ("OK\n");
}
