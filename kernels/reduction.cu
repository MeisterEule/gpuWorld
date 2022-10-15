#include <stdio.h>

#include "compute_step.h"
#include "grid_utils.h"

#define BLOCK_DIM 1024

__global__ void segmented_sum_reduction_kernel (int *input, int *output) {
	extern __shared__ int input_s[];

	int segment = 2 * blockDim.x * blockIdx.x;
	int gtid = segment + threadIdx.x;
	int ltid = threadIdx.x;

	input_s[ltid] = input[gtid] + input[gtid + blockDim.x];
	__syncthreads();

	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		__syncthreads();
		if (ltid < stride) input_s[ltid] += input_s[ltid + stride];
	}
        if (ltid == 0) atomicAdd(&(output[0]), input_s[0]);
}

int computeArraySum (compute_step_t *cs_h) {
	cs_pad (cs_h, 2 * BLOCK_DIM);

	int n_threads, n_blocks;
	getGridDimension1D (cs_h->n_data_in, &n_blocks, &n_threads);


	int *data_d;
	if (cs_h->input_on_device) {
		data_d = cs_h->data_in;
	} else {
		cudaMalloc((void**)&data_d, cs_h->n_data_in * sizeof(int));
		cudaMemcpy(data_d, cs_h->data_in, cs_h->n_data_in * sizeof(int), cudaMemcpyHostToDevice);
	}
	int *sum_d;
	if (cs_h->output_on_device) {
		sum_d = cs_h->data_out;
	} else {
		cudaMalloc((void**)&sum_d, cs_h->n_data_out * sizeof(int));
		cudaMemset(sum_d, 0, cs_h->n_data_out * sizeof(int));
	}

	segmented_sum_reduction_kernel<<<n_blocks,n_threads,n_threads * sizeof(int)>>>(data_d, sum_d);
	int sum;
	cudaMemcpy (&sum, sum_d, sizeof(int), cudaMemcpyDeviceToHost);
	if (!cs_h->input_on_device) cudaFree(data_d);
	return sum;
}
