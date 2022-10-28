#include <stdio.h>
#include <math.h>

#include "memoryManager.hpp"
#include "spmv.hpp"
#include "random.h"
#include "count.h"

#define NONZERO_RATIO 0.5

bool checkCPU (float *M, float *v, float *v_ref, int Nrows) {
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
	int N = argc > 1 ? atoi(argv[1]) : 4;
	unsigned long long n_matrix_elements = (unsigned long long)N * N;
	printf ("n_matrix_elements: %lld\n", n_matrix_elements);

	memoryManager *mm = new memoryManager(true);
	cudaRNG *rng = new cudaRNG (1024 * 1024, DEFAULT_SEED);

	rng->initRNG (mm, N);
	float *matrix = rng->generateRandomMatrix (mm, n_matrix_elements, NONZERO_RATIO);	
	float *vector = rng->generateRandomMatrix (mm, N, 1.0);
	rng->freeRNG(mm);

	//printf ("matrix: \n");
	//for (int row = 0; row < N; row++) {
	//	for (int col = 0; col < N; col++) {
	//		printf ("%5.3f ", matrix[row * N + col]);
	//	}
	//	printf ("\n");
	//}
	//printf ("input vector: \n");
	//for (int i = 0; i < N; i++) {
	//	printf ("%5.3f\n", vector[i]);
	//}

	float *result_gpu = spMVCoo<float> (mm, matrix, vector, N);

	//printf ("output vector: \n");
	//for (int i = 0; i < N; i++) {
	//	printf ("%5.3f\n", result_gpu[i]);
	//}

	printf ("Okay: %d\n", checkCPU (matrix, vector, result_gpu, N));
	return 0;
}

