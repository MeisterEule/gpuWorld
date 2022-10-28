#include "types.h"
#include "memoryManager.hpp"
#include "spmv.hpp"
#include "random.h"
#include "count.h"

#define NONZERO_RATIO 0.5

int main (int argc, char *argv[]) {
	LDIM N = argc > 1 ? atoll(argv[1]) : 4;
	if (N > 1024) {
		printf ("This test only works with N < 1024!\n");
		return -1;
	}
	LDIM n_matrix_elements = N * N;
	printf ("n_matrix_elements: %lld\n", n_matrix_elements);

	memoryManager *mm = new memoryManager(true);
	cudaRNG *rng = new cudaRNG (1024 * 1024, DEFAULT_SEED);

	rng->initRNG (mm, n_matrix_elements);
	float *matrix = rng->generateRandomMatrix (mm, n_matrix_elements, NONZERO_RATIO);	
	float *vector = rng->generateRandomMatrix (mm, N, 1.0);
	rng->freeRNG(mm);

	LDIM nnz = countNonzeros (mm, matrix, n_matrix_elements, false);
	csrMatrix<float,int> csr_matrix (mm, matrix, N, nnz);
}
