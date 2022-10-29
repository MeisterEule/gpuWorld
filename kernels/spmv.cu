#include "types.h"
#include "memoryManager.hpp"
#include "grid_utils.h"
#include "spmv.hpp"

#include "random.h"

__global__ void spmv_coo_kernel (int *rowidx, int *colidx, float *values, float *v, float *w, LDIM nnz) {
 	LDIM idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= nnz) return;

	int row = rowidx[idx];
	int col = colidx[idx];
	float value = values[idx];
	atomicAdd(&w[row], v[col]*value);
}

__global__ void spmv_coo_kernel (LDIM *rowidx, LDIM *colidx, float *values, float *v, float *w, LDIM nnz) {
 	LDIM idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= nnz) return;

	LDIM row = rowidx[idx];
	LDIM col = colidx[idx];
	float value = values[idx];
	atomicAdd(&w[row], v[col]*value);
}

__global__ void spmv_csr_kernel (LDIM *rowptrs, LDIM *colidx, float *values, float *v, float *w, LDIM nnz) {
	LDIM row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row >= nnz) return;

	float sum = 0;
	for (LDIM i = rowptrs[row]; i < rowptrs[row+1]; i++) {
	   LDIM col = colidx[i];
	   sum += values[i] * v[col];
	}
	w[row] += sum;
}

__global__ void spmv_csr_kernel (int *rowptrs, int *colidx, float *values, float *v, float *w, LDIM nnz) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row >= nnz) return;

	float sum = 0;
	for (int i = rowptrs[row]; i < rowptrs[row+1]; i++) {
	   int col = colidx[i];
	   sum += values[i] * v[col];
	}
	w[row] += sum;
}

__global__ void smpv_csr_kernel (LDIM *rowptr, LDIM *colidx, float *values, float *v, float *w, LDIM nnz) {
}

void launch_spmv_coo_kernel (int n_blocks, int n_threads, int *rowidx, int *colidx, float *values, float *v, float *w, LDIM nnz) {
	spmv_coo_kernel<<<n_blocks,n_threads>>>(rowidx, colidx, values, v, w, nnz);
}

void launch_spmv_coo_kernel (int n_blocks, int n_threads, LDIM *rowidx, LDIM *colidx, float *values, float *v, float *w, LDIM nnz) {
	spmv_coo_kernel<<<n_blocks,n_threads>>>(rowidx, colidx, values, v, w, nnz);
}

void launch_spmv_csr_kernel (int n_blocks, int n_threads, int *rowptr, int *colidx, float *values, float *v, float *w, LDIM nnz) {
	spmv_csr_kernel<<<n_blocks,n_threads>>>(rowptr, colidx, values, v, w, nnz);
}


void launch_spmv_csr_kernel (int n_blocks, int n_threads, LDIM *rowptr, LDIM *colidx, float *values, float *v, float *w, LDIM nnz) {
	spmv_csr_kernel<<<n_blocks,n_threads>>>(rowptr, colidx, values, v, w, nnz);
}

