#include "memoryManager.hpp"
//#include "compute_step.hpp"
#include "spmv.hpp"

#include "random.h"

__global__ void spmv_coo_kernel (int *rowidx, int *colidx, float *values, float *v, float *w, long long  nnz) {
 	long long idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= nnz) return;

	int row = rowidx[idx];
	int col = colidx[idx];
	float value = values[idx];
	atomicAdd(&w[row], v[col]*value);
}


template <typename T> T *spMVCoo (memoryManager *mm, T *matrix, T *v_in, int n_elements) {
	T *v_out = (T*)malloc(n_elements * sizeof(T));
	memset (v_out, 0, n_elements * sizeof(T));

	long long nnz = countNonzeros (mm, matrix, n_elements, false);

	cooMatrix<T> coo_matrix (matrix, n_elements, nnz);

	int *rowidx_d, *colidx_d, *values_d;

	mm->deviceAllocate<int> (rowidx_d, nnz, "coorowidx");
	mm->deviceAllocate<int> (colidx_d, nnz, "coocolidx");
	mm->deviceAllocate<T> (values_d, nnz, "coovalues");

	cudaMemcpy (rowidx_d, coo_matrix.rowidx, nnz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy (colidx_d, coo_matrix.colidx, nnz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy (values_d, coo_matrix.values, nnz * sizeof(T), cudaMemcpyHostToDevice);
	
	T *v_in_d, *v_out_d;
	mm->deviceAllocate<T> (v_in_d, n_elements, "inputVector");
	mm->deviceAllocate<T> (v_out_d, n_elements, "outputVector");

	cudaMemcpy (v_in_d, v_in, n_elements * sizeof(T), cudaMemcpyHostToDevice);
	cudaMemcpy (v_out_d, v_out, n_elements * sizeof(T), cudaMemcpyHostToDevice);
   
	int n_threads, n_blocks;
    	get_grid_dimension (nnz, &n_blocks, &n_threads);

	spmv_coo_kernel<<<n_blocks,n_threads>>>(rowidx_d, colidx_d, values_d, v_in_d, v_out_d, nnz);

}

