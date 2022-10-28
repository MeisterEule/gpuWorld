#include <stdio.h>
#include <math.h>
#include <limits.h>

#include "types.h"
#include "memoryManager.hpp"
#include "spmv.hpp"
#include "random.h"
#include "count.h"

#define NONZERO_RATIO 0.5

bool checkCPU (float *M, float *v, float *v_ref, LDIM Nrows) {
	float *v_res = (float*)malloc(Nrows * sizeof(float));
	for (int row = 0; row < Nrows; row++) {
		v_res[row] = 0;
		for (int col = 0; col < Nrows; col++) {
			v_res[row] += M[row * Nrows + col] * v[col];	
		}
	}

	for (int i = 0; i < Nrows; i++) {
		if (fabs(v_ref[i] - v_res[i]) > 0.001) return false;
	}
	return true;
}

int main (int argc, char *argv[]) {
	LDIM N = argc > 1 ? atoll(argv[1]) : 4;
	LDIM n_matrix_elements = N * N;
	printf ("n_matrix_elements: %lld\n", n_matrix_elements);

	memoryManager *mm = new memoryManager(true);
	cudaRNG *rng = new cudaRNG (1024 * 1024, DEFAULT_SEED);

	rng->initRNG (mm, N);
	float *matrix = rng->generateRandomMatrix (mm, n_matrix_elements, NONZERO_RATIO);	
	float *vector = rng->generateRandomMatrix (mm, N, 1.0);
	rng->freeRNG(mm);


	LDIM nnz = countNonzeros (mm, matrix, n_matrix_elements, false);
	printf ("Nr. of nonzeros: %lld (%f)\n", nnz, (float)nnz / n_matrix_elements);

	float *result_gpu;
	if (nnz < INT_MAX) {
		printf ("Use small spmv\n");
		result_gpu = spMVCoo<float,int> (mm, matrix, vector, N, nnz);
	} else {
		printf ("Use large spmv\n");
		result_gpu = spMVCoo<float,LDIM> (mm, matrix, vector, N, nnz);
	}

	printf ("Okay: %d\n", checkCPU (matrix, vector, result_gpu, N));
	return 0;
}

