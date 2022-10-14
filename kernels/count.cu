#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "grid_utils.h"

#define SAFE_CUDA(apiFuncCall)                                          \
do {                                                                           \
    cudaError_t _status = apiFuncCall;                                         \
    if (_status != cudaSuccess) {                                              \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",   \
                __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status));\
        exit(-1);                                                              \
    } \
} while (0)


__global__ void count_elements_in_array_kernel_int (int *data, int *count, int n_data) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= n_data) return;
	atomicAdd(&(count[data[tid]]), 1);
}

__global__ void count_elements_in_array_kernel (char *data, int *count, int n_data) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= n_data) return;
	int alphabet_position = data[tid] - 'a';
	if (alphabet_position >= 0 && alphabet_position < 26) {
	   atomicAdd(&(count[alphabet_position]), 1);
	}
}


int *countElementsInArray (int *data, int n_data) {
	printf ("HUHU\n");
	int n_threads, n_blocks;
	getGridDimension1D (n_data, &n_blocks, &n_threads);
        int mem_size = n_data * sizeof(int);

	int *count_h = (int*)malloc(mem_size);
	memset (count_h, 0, mem_size);

	int *data_d, *count_d;
	SAFE_CUDA(cudaMalloc((void**)&data_d, mem_size));
	SAFE_CUDA(cudaMalloc((void**)&count_d, mem_size));
	SAFE_CUDA(cudaMemcpy(data_d, data, mem_size, cudaMemcpyHostToDevice));
	SAFE_CUDA(cudaMemset(count_d, 0, mem_size));

	count_elements_in_array_kernel_int<<<n_blocks,n_threads>>>(data_d, count_d, n_data);
	cudaError_t ce = cudaGetLastError();
	printf ("cudaError: %s\n", cudaGetErrorString(ce));

	SAFE_CUDA(cudaMemcpy(count_h, count_d, mem_size, cudaMemcpyDeviceToHost));
	SAFE_CUDA(cudaFree(count_d));
	SAFE_CUDA(cudaFree(data_d));

	return count_h;
}

int *countElementsInArray (char *data, int n_data) {
	int n_threads, n_blocks;
	getGridDimension1D (n_data, &n_blocks, &n_threads);

	int *count_h = (int*)malloc(n_data * sizeof(int));
	memset (count_h, 0, 26 * sizeof(int));

	char *data_d;
        int *count_d;
	cudaMalloc((void**)&data_d, n_data * sizeof(char));
	cudaMalloc((void**)&count_d, 26 * sizeof(int));
	cudaMemcpy(data_d, data, n_data * sizeof(char), cudaMemcpyHostToDevice);
	cudaMemset(count_d, 0, n_data * sizeof(int));

	count_elements_in_array_kernel<<<n_blocks,n_threads>>>(data_d, count_d, n_data);

	cudaMemcpy(count_d, count_h, n_data * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(count_d);
	cudaFree(data_d);

	return count_h;
}
