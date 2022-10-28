#ifndef SPMV_HPP
#define SPMV_HPP

#include "memoryManager.hpp"
#include "grid_utils.h"
#include "random.h"
//#include "cuda_launcher.cuh"

template<typename T> class cooMatrix
{
	public:
		long long nnz;
		int *rowidx;
		int *colidx;
		T *values;
	
		cooMatrix (T *MSimple, int Nrows, long long nnz);
};

template<typename T> class csrMatrix
{
	public:
		long long nnz;
		int *rowptr;
		int *colidx;
		T *values;

		//csrMatrix (cooMatrix *cm, int n_elements);
};

template<typename T> cooMatrix<T>::cooMatrix(T *MSimple, int Nrows, long long nnz) {
	nnz = nnz;
	rowidx = (int*)malloc(nnz * sizeof(int));
	colidx = (int*)malloc(nnz * sizeof(int));
	values = (T*)malloc(nnz * sizeof(T));

	int idx = 0;
	for (int row = 0; row < Nrows; row++) {
		for (int col = 0; col < Nrows; col++) {
			//printf ("index: %d\n", row * n_elements + col);
			//printf ("idx: %d\n", 
			if (MSimple[row * Nrows + col] != 0) {
				rowidx[idx] = row;
				colidx[idx] = col;
				values[idx] = MSimple[row * Nrows + col];
				idx++;
			}
		}
	}
}

extern void launch_spmv_coo_kernel (int n_blocks, int n_threads, int *rowidx, int *colidx, float *values, float *v, float *w, long long nnz);


template <typename T> T *spMVCoo (memoryManager *mm, T *matrix, T *v_in, int Nrows) {
	T *v_out = (T*)malloc(Nrows * sizeof(T));
	memset (v_out, 0, Nrows * sizeof(T));

	long long nnz = countNonzeros (mm, matrix, (unsigned long long)Nrows * Nrows, false);
	printf ("Nr. of nonzeros: %lld\n", nnz);

	cooMatrix<T> coo_matrix (matrix, Nrows, nnz);

        T *values_d;
	int *rowidx_d, *colidx_d;

	mm->deviceAllocate<int> (rowidx_d, nnz, "coorowidx");
	mm->deviceAllocate<int> (colidx_d, nnz, "coocolidx");
	mm->deviceAllocate<T> (values_d, nnz, "coovalues");

	cudaMemcpy (rowidx_d, coo_matrix.rowidx, nnz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy (colidx_d, coo_matrix.colidx, nnz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy (values_d, coo_matrix.values, nnz * sizeof(T), cudaMemcpyHostToDevice);
	
	T *v_in_d, *v_out_d;
	mm->deviceAllocate<T> (v_in_d, Nrows, "inputVector");
	mm->deviceAllocate<T> (v_out_d, Nrows, "outputVector");

	cudaMemcpy (v_in_d, v_in, Nrows * sizeof(T), cudaMemcpyHostToDevice);
	cudaMemcpy (v_out_d, v_out, Nrows * sizeof(T), cudaMemcpyHostToDevice);
   
	int n_threads, n_blocks;
    	getGridDimension1D (nnz, &n_blocks, &n_threads);

	launch_spmv_coo_kernel (n_blocks, n_threads, rowidx_d, colidx_d, values_d, v_in_d, v_out_d, nnz);

	cudaMemcpy (v_out, v_out_d, Nrows * sizeof(T), cudaMemcpyDeviceToHost);

	mm->deviceFree<int> (rowidx_d);
	mm->deviceFree<int> (colidx_d);
	mm->deviceFree<T> (values_d);
	mm->deviceFree<T> (v_in_d);
	mm->deviceFree<T> (v_out_d);

	return v_out;
}

#endif
