#ifndef SPMV_HPP
#define SPMV_HPP

#include "types.h"
#include "memoryManager.hpp"
#include "grid_utils.h"
#include "random.h"

template<typename T,typename U> class cooMatrix
{
	public:
		LDIM nnz;
		U *rowidx;
		U *colidx;
		T *values;
	
		cooMatrix (T *MSimple, LDIM Nrows, LDIM nnz);
};

template<typename T,typename U> class csrMatrix
{
	public:
		LDIM nnz;
		U *rowptr;
		U *colidx;
		T *values;

		//csrMatrix (cooMatrix *cm, int n_elements);
};

template<typename T,typename U> cooMatrix<T,U>::cooMatrix(T *MSimple, LDIM Nrows, LDIM nnz) {
	nnz = nnz;
	rowidx = (U*)malloc(nnz * sizeof(U));
	colidx = (U*)malloc(nnz * sizeof(U));
	values = (T*)malloc(nnz * sizeof(T));

	LDIM idx = 0;
	for (LDIM row = 0; row < Nrows; row++) {
		for (LDIM col = 0; col < Nrows; col++) {
			if (MSimple[row * Nrows + col] != 0) {
				rowidx[idx] = (U)row;
				colidx[idx] = (U)col;
				values[idx] = MSimple[row * Nrows + col];
				idx++;
			}
		}
	}
}

extern void launch_spmv_coo_kernel (int n_blocks, int n_threads, int *rowidx, int *colidx, float *values, float *v, float *w, LDIM nnz);
extern void launch_spmv_coo_kernel (int n_blocks, int n_threads, LDIM *rowidx, LDIM *colidx, float *values, float *v, float *w, LDIM nnz);


template <typename T, typename U> T *spMVCoo (memoryManager *mm, T *matrix, T *v_in, LDIM Nrows, LDIM nnz) {
	printf ("is possible? %d\n", mm->isPossible<U> (3 * nnz));

	T *v_out = (T*)malloc(Nrows * sizeof(T));
	memset (v_out, 0, Nrows * sizeof(T));

	cooMatrix<T,U> coo_matrix (matrix, Nrows, nnz);

        T *values_d;
	U *rowidx_d, *colidx_d;

	mm->deviceAllocate<U> (rowidx_d, nnz, "coorowidx");
	mm->deviceAllocate<U> (colidx_d, nnz, "coocolidx");
	mm->deviceAllocate<T> (values_d, nnz, "coovalues");

	cudaMemcpy (rowidx_d, coo_matrix.rowidx, nnz * sizeof(U), cudaMemcpyHostToDevice);
	cudaMemcpy (colidx_d, coo_matrix.colidx, nnz * sizeof(U), cudaMemcpyHostToDevice);
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

	mm->deviceFree<U> (rowidx_d);
	mm->deviceFree<U> (colidx_d);
	mm->deviceFree<T> (values_d);
	mm->deviceFree<T> (v_in_d);
	mm->deviceFree<T> (v_out_d);

	return v_out;
}

#endif
