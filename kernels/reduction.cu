#include <stdio.h>

#include "compute_step.hpp"
#include "memoryManager.hpp"
#include "grid_utils.h"

#define BLOCK_DIM 1024

__global__ void segmented_sum_reduction_kernel (int *input, int *output) {
	extern __shared__ int input_s[];

	int segment = 2 * blockDim.x * blockIdx.x;
	int gtid = segment + threadIdx.x;
	int ltid = threadIdx.x;

	input_s[ltid] = input[gtid] + input[gtid + blockDim.x];
	__syncthreads();

	for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
		__syncthreads();
		if (ltid < stride) input_s[ltid] += input_s[ltid + stride];
	}
        if (ltid == 0) atomicAdd(&(output[0]), input_s[0]);
}

int computeArraySum (memoryManager *mm, ComputeStep<int> *cs_h) {
	cs_h->Pad(2 * BLOCK_DIM);

	int n_data_in = cs_h->n_data_in->front();
        int n_data_out = cs_h->n_data_out->front();
	int *data_in = cs_h->data_in->front();
	int *data_out = cs_h->data_out->front(); 
	bool input_on_device = cs_h->input_on_device->front();
	bool output_on_device = cs_h->output_on_device->front();

	int n_threads = GRID_MAX_THREADS / 2;
	int n_blocks = n_data_in / n_threads / 2;


	int *data_d;
	if (input_on_device) {
		data_d = data_in;
	} else {
		//cudaMalloc((void**)&data_d, n_data_in * sizeof(int));
		mm->deviceAllocate(data_d, n_data_in);
		cudaMemcpy(data_d, data_in, n_data_in * sizeof(int), cudaMemcpyHostToDevice);
	}
	int *sum_d;
	if (output_on_device) {
		sum_d = data_out;
	} else {
		//cudaMalloc((void**)&sum_d, n_data_out * sizeof(int));
	 	mm->deviceAllocate(sum_d, n_data_out);
		cudaMemset(sum_d, 0, n_data_out * sizeof(int));
	}

	segmented_sum_reduction_kernel<<<n_blocks,n_threads,n_threads * sizeof(int)>>>(data_d, sum_d);
	int sum;
	cudaMemcpy (&sum, sum_d, sizeof(int), cudaMemcpyDeviceToHost);
	if (!input_on_device) cudaFree(data_d);
	return sum;
}
