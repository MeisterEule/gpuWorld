#ifndef SPMV_HPP
#define SPMV_HPP

template<typename T> class cooMatrix
{
	public:
		long long nnz;
		int *rowidx;
		int *colidx;
		T *values;
	
		cooMatrix (T *MSimple, int n_elements, long long nnz);
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

template<typename T> cooMatrix<T>::cooMatrix(T *MSimple, int n_elements, long long nnz) {
	nnz = nnz;
	rowidx = (int*)malloc(nnz * sizeof(int));
	colidx = (int*)malloc(nnz * sizeof(int));
	values = (T*)malloc(nnz * sizeof(T));

	int idx = 0;
	for (int row = 0; row < n_elements; row++) {
		for (int col = 0; col < n_elements; col++) {
			if (MSimple[row * n_elements + col] != 0) {
				rowidx[idx] = row;
				colidx[idx] = col;
				values[idx] = MSimple[row * n_elements + col];
				idx++;
			}
		}
	}
}

#endif
