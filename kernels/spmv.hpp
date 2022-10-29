#ifndef SPMV_HPP
#define SPMV_HPP

#include "types.h"
#include "memoryManager.hpp"
#include "grid_utils.h"
#include "random.h"
#include "count.h"
#include "scan.h"

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

		csrMatrix (memoryManager *mm, T *Msimple, LDIM Nrows, LDIM nnz);
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

template<typename T, typename U> csrMatrix<T,U>::csrMatrix (memoryManager *mm, T *MSimple, LDIM Nrows, LDIM nnz) {
	nnz = nnz;
	U *rowidx = (U*)malloc(nnz * sizeof(U));
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

	U *rowidx_d, *count_d, *scan_d;

	U *count_h = (U*)malloc(Nrows * sizeof(U));
	//U *scan_h = (U*)malloc(Nrows * sizeof(U));
	rowptr = (U*)malloc(Nrows * sizeof(U));

	countElementsInArray (mm, rowidx, count_h, nnz, Nrows, false, false);
	//scanArray (mm, count_h, scan_h, Nrows, Nrows, false, false, true);
	scanArray (mm, count_h, rowptr, Nrows, Nrows, false, false, true);

	//printf ("rowidx: \n");
	//for (int i = 0; i < nnz; i++) {
	//	printf ("%d(%d) ", rowidx[i], i);
	//}
	//printf ("\n");

	//printf ("colidx: \n");
	//for (int i = 0; i < nnz; i++) {
	//	printf ("%d ", colidx[i]);
	//}
	//printf ("\n");

	//printf ("values: \n");
	//for (int i = 0; i < nnz; i++) {
	//	printf ("%f ", values[i]);
	//}
	//printf ("\n");

	//printf ("rowptr: \n");
	//for (int i = 0; i < Nrows; i++) {
	//	printf ("%d ", scan_h[i]);
	//}
	//printf ("\n");

	free (count_h);
	//free (scan_h);

}

extern void launch_spmv_coo_kernel (int n_blocks, int n_threads, int *rowidx, int *colidx, float *values, float *v, float *w, LDIM nnz);
extern void launch_spmv_coo_kernel (int n_blocks, int n_threads, LDIM *rowidx, LDIM *colidx, float *values, float *v, float *w, LDIM nnz);

extern void launch_spmv_csr_kernel (int n_blocks, int n_threads, int *rowptr, int *colidx, float *values, float *v, float *w, LDIM nnz);
extern void launch_spmv_csr_kernel (int n_blocks, int n_threads, LDIM *rowptr, LDIM *colidx, float *values, float *v, float *w, LDIM nnz);

template <typename T, typename U> T *spMVCoo (memoryManager *mm, T *matrix, T *v_in, LDIM Nrows, LDIM nnz) {

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

template <typename T, typename U> T* spMVCsr (memoryManager *mm, T *matrix, T *v_in, LDIM Nrows, LDIM nnz) {
	T *v_out = (T*)malloc(Nrows * sizeof(T));
	memset (v_out, 0, Nrows * sizeof(T));

	csrMatrix<T,U> csr_matrix (mm, matrix, Nrows, nnz);

        T *values_d;
	U *rowptr_d, *colidx_d;

	//printf ("Allocate: %d %d\n", Nrows, nnz);
	//cudaMalloc((void**)&rowptr_d, Nrows * sizeof(U));
	//cudaMAlloc((void**)&colidx_d, nnz * sizeof(U));
	//cudaMAlloc((void**)&values_d, nnz * sizeof(T));
	mm->deviceAllocate<U> (rowptr_d, Nrows, "csrrowptr");
	mm->deviceAllocate<U> (colidx_d, nnz, "csrcolidx");
	mm->deviceAllocate<T> (values_d, nnz, "csrvalues");

	//printf ("Check 1\n");
	//fflush(stdout);
	cudaMemcpy (rowptr_d, csr_matrix.rowptr, Nrows * sizeof(U), cudaMemcpyHostToDevice);
	//printf ("Check 1.1\n");
	//fflush(stdout);
	cudaMemcpy (colidx_d, csr_matrix.colidx, nnz * sizeof(U), cudaMemcpyHostToDevice);
	//printf ("Check 1.2\n");
	//fflush(stdout);
	cudaMemcpy (values_d, csr_matrix.values, nnz * sizeof(T), cudaMemcpyHostToDevice);
	//printf ("Check 2\n");
	//fflush(stdout);
	
	T *v_in_d, *v_out_d;
	mm->deviceAllocate<T> (v_in_d, Nrows, "CSRinputVector");
	mm->deviceAllocate<T> (v_out_d, Nrows, "CSRoutputVector");

	cudaMemcpy (v_in_d, v_in, Nrows * sizeof(T), cudaMemcpyHostToDevice);
	cudaMemcpy (v_out_d, v_out, Nrows * sizeof(T), cudaMemcpyHostToDevice);
   
	int n_threads, n_blocks;
    	getGridDimension1D (nnz, &n_blocks, &n_threads);

	launch_spmv_csr_kernel (n_blocks, n_threads, rowptr_d, colidx_d, values_d, v_in_d, v_out_d, nnz);

	cudaMemcpy (v_out, v_out_d, Nrows * sizeof(T), cudaMemcpyDeviceToHost);

	mm->deviceFree<U> (rowptr_d);
	mm->deviceFree<U> (colidx_d);
	mm->deviceFree<T> (values_d);
	mm->deviceFree<T> (v_in_d);
	mm->deviceFree<T> (v_out_d);

	return v_out;

}

#endif
