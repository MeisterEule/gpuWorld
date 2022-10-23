#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "compute_step.hpp"
#include "memoryManager.hpp"
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

__global__ void count_elements_in_array_kernel_char (char *data, int *count, int n_data) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= n_data) return;
	if (data[tid] >= 0 && data[tid] < 128) {
	   atomicAdd(&(count[data[tid]]), 1);
	}
}

__global__ void avg_atomic_kernel (int *data, int *count, int n_data) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= n_data) return;
	atomicAdd(&(count[0]), data[tid]);
	__syncthreads();
}


void countElementsInArray (memoryManager *mm, ComputeStep<int,int> *cs_h) {
        int n_data_in = cs_h->n_data_in->front();
	int n_data_out = cs_h->n_data_out->front();
        bool input_on_device = cs_h->input_on_device->front();
	bool output_on_device = cs_h->input_on_device->front();
	int *data_in = cs_h->data_in->front();
        int *data_out = cs_h->data_out->front();
	int n_threads, n_blocks;
	getGridDimension1D (n_data_in, &n_blocks, &n_threads);


	int *data_d;
	if (input_on_device) {
		data_d = data_in;
	} else {
		mm->deviceAllocate<int>(data_d, n_data_in, "countInput");
	 	cudaMemcpy(data_d, data_in, n_data_in * sizeof(int), cudaMemcpyHostToDevice);
	}

	int *count_d;
	if (output_on_device) {
		count_d = data_out;
	} else {
		mm->deviceAllocate<int>(count_d, n_data_out, "countOutput");
	}
	cudaMemset(count_d, 0, n_data_out * sizeof(int));

	count_elements_in_array_kernel_int<<<n_blocks,n_threads>>>(data_d, count_d, n_data_in);

	if (!output_on_device) {
	   cudaMemcpy(data_out, count_d, n_data_out * sizeof(int), cudaMemcpyDeviceToHost);
	}
	if (!input_on_device) mm->deviceFree(data_d, (uint64_t)&data_d);
	if (!output_on_device) mm->deviceFree(count_d, (uint64_t)&count_d);
}

int *countElementsInArray (memoryManager *mm, ComputeStep<char,int> *cs_h) {
        int n_data_in = cs_h->n_data_in->front();
	int n_data_out = cs_h->n_data_out->front();
	if (n_data_out != 128) {
		printf ("Character count output must have 128 elements (Nr. of symbols in ASCII).\n");
		return NULL;
	}
        bool input_on_device = cs_h->input_on_device->front();
	bool output_on_device = cs_h->input_on_device->front();
	char *data_in = cs_h->data_in->front();
        int *data_out = cs_h->data_out->front();
	int n_threads, n_blocks;
	getGridDimension1D (n_data_in, &n_blocks, &n_threads);

	char *data_d;
	if (input_on_device) {
		data_d = data_in;
	} else {
		mm->deviceAllocate<char>(data_d, n_data_in, "countInput");
	 	cudaMemcpy(data_d, data_in, n_data_in * sizeof(char), cudaMemcpyHostToDevice);
	}

	int *count_d;
	if (output_on_device) {
		count_d = data_out;
	} else {
		mm->deviceAllocate<int>(count_d, n_data_out, "countOutput");
	}
	cudaMemset(count_d, 0, n_data_out * sizeof(int));

	count_elements_in_array_kernel_char<<<n_blocks,n_threads>>>(data_d, count_d, n_data_in);

	int *count_h = (int*)malloc(26 * sizeof(int));

	if (!output_on_device) cudaMemcpy(count_h, count_d, n_data_out * sizeof(int), cudaMemcpyDeviceToHost);
	if (!input_on_device) mm->deviceFree(count_d, (uint64_t)&count_d);
	if (!output_on_device) mm->deviceFree(data_d, (uint64_t)&data_d);

	return count_h;
}
