#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "compute_step.h"
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

__global__ void avg_atomic_kernel (int *data, int *count, int n_data) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= n_data) return;
	atomicAdd(&(count[0]), data[tid]);
	__syncthreads();
}


void countElementsInArray (compute_step_t *cs_h) {
	int n_threads, n_blocks;
	getGridDimension1D (cs_h->n_data_in, &n_blocks, &n_threads);


	int *data_d;
	if (cs_h->input_on_device) {
		data_d = cs_h->data_in;
	} else {
		cudaMalloc((void**)&data_d, cs_h->n_data_in * sizeof(int));
	 	cudaMemcpy(data_d, cs_h->data_in, cs_h->n_data_in * sizeof(int), cudaMemcpyHostToDevice);
	}

	int *count_d;
	if (cs_h->output_on_device) {
		count_d = cs_h->data_out;
	} else {
		cudaMalloc((void**)&count_d, cs_h->n_data_out * sizeof(int));
	}
	cudaMemset(count_d, 0, cs_h->n_data_out * sizeof(int));

	count_elements_in_array_kernel_int<<<n_blocks,n_threads>>>(data_d, count_d, cs_h->n_data_in);

	if (!cs_h->output_on_device) {
	   cudaMemcpy(cs_h->data_out, count_d, cs_h->n_data_out * sizeof(int), cudaMemcpyDeviceToHost);
	}
	if (!cs_h->input_on_device) cudaFree(data_d);
	if (!cs_h->output_on_device) cudaFree(count_d);
}

void computeAverageOfArray (compute_step_t *cs_h) {
	// TODO: Check if N_out = 1
	int n_threads, n_blocks;
	getGridDimension1D (cs_h->n_data_in, &n_blocks, &n_threads);
 	if (n_blocks > 1) printf ("Warning: Only one block for reduction supported\n");	

	int *data_d;
	if (cs_h->input_on_device) {
		data_d = cs_h->data_in;
	} else {
		cudaMalloc((void**)&data_d, cs_h->n_data_in * sizeof(int));
		cudaMemcpy(data_d, cs_h->data_in, cs_h->n_data_in * sizeof(int), cudaMemcpyHostToDevice);
	}

	int *count_d;
	if (cs_h->output_on_device) {
		count_d = cs_h->data_out;
	} else {
		cudaMalloc((void**)&count_d, cs_h->n_data_out * sizeof(int));
	}
	cudaMemset(count_d, 0, cs_h->n_data_out * sizeof(int));

	avg_atomic_kernel<<<n_blocks,n_threads>>>(data_d, count_d, cs_h->n_data_in);

	if (!cs_h->output_on_device) {
	   cudaMemcpy(cs_h->data_out, count_d, cs_h->n_data_out * sizeof(int), cudaMemcpyDeviceToHost);
	}
	if (!cs_h->input_on_device) cudaFree(data_d);
	if (!cs_h->output_on_device) cudaFree(count_d);
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
