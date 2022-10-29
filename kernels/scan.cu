#include "compute_step.hpp"
#include "memoryManager.hpp"
#include "grid_utils.h"

#define BLOCK_DIM 1024

__global__ void scan_kernel (int *in, int *out, int n_elements, bool exclusive) {
	extern __shared__ int inout[];
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	// Fill shared memory
	if (exclusive) {
		if (tid < n_elements && threadIdx.x > 0) {
			inout[threadIdx.x] = in[tid - 1];
		} else {
			inout[threadIdx.x] = 0;
		}
	} else {
		if (tid < n_elements) {
			inout[threadIdx.x] = in[tid];
		} else {
			inout[threadIdx.x] = 0;
		}
	}

	for (int stride = 1; stride < blockDim.x; stride *=2) {
		__syncthreads();
		int tmp;
		if (threadIdx.x >= stride) {
			tmp = inout[threadIdx.x] + inout[threadIdx.x - stride];
		}
		__syncthreads();
		if (threadIdx.x >= stride) inout[threadIdx.x] = tmp;
	}
	if (tid < n_elements) out[tid] = inout[threadIdx.x];
}

void scanArray (memoryManager *mm, int *data_in, int *data_out, int n_data_in, int n_data_out,
		bool input_on_device, bool output_on_device, bool exclusive) {
 	int n_threads, n_blocks;
	getGridDimension1D (n_data_in, &n_blocks, &n_threads);

	/// THIS SIMPLE SCAN ONLY WORKS FOR ONE BLOCK!

	int *data_d;
	if (input_on_device) {
		data_d = data_in;
	} else {
		mm->deviceAllocate<int>(data_d, n_data_in, "countInput");
	 	cudaMemcpy(data_d, data_in, n_data_in * sizeof(int), cudaMemcpyHostToDevice);
	}

	int *scan_d;
	if (output_on_device) {
		scan_d = data_out;
	} else {
		mm->deviceAllocate<int>(scan_d, n_data_out, "countOutput");
	}
	cudaMemset(scan_d, 0, n_data_out * sizeof(int));
	scan_kernel<<<n_blocks,n_threads,n_blocks*n_threads*sizeof(int)>>>(data_d, scan_d, n_data_in, exclusive);

	if (!output_on_device) cudaMemcpy (data_out, scan_d, n_data_out * sizeof(int), cudaMemcpyDeviceToHost);
	if (!input_on_device) mm->deviceFree<int>(data_d);
	if (!output_on_device) mm->deviceFree<int>(scan_d);
}


void scanArray (memoryManager *mm, ComputeStep<int,int> *cs, bool exclusive) {
	//cs->Pad (2 * BLOCK_DIM);
	int n_data_in = cs->n_data_in->front();
	int n_data_out = cs->n_data_out->front();
        bool input_on_device = cs->input_on_device->front();
	bool output_on_device = cs->input_on_device->front();
	int *data_in = cs->data_in->front();
        int *data_out = cs->data_out->front();
	scanArray (mm, data_in, data_out, n_data_in, n_data_out, input_on_device, output_on_device, exclusive);
	//int n_threads, n_blocks;
	//getGridDimension1D (n_data_in, &n_blocks, &n_threads);

	/// THIS SIMPLE SCAN ONLY WORKS FOR ONE BLOCK!

	//int *data_d;
	//if (input_on_device) {
	//	data_d = data_in;
	//} else {
	//	mm->deviceAllocate<int>(data_d, n_data_in, "countInput");
	// 	cudaMemcpy(data_d, data_in, n_data_in * sizeof(int), cudaMemcpyHostToDevice);
	//}

	//int *scan_d;
	//if (output_on_device) {
	//	scan_d = data_out;
	//} else {
	//	mm->deviceAllocate<int>(scan_d, n_data_out, "countOutput");
	//}
	//cudaMemset(scan_d, 0, n_data_out * sizeof(int));
	//scan_kernel<<<n_blocks,n_threads,n_blocks*n_threads*sizeof(int)>>>(data_d, scan_d, n_data_in);

	//if (!output_on_device) cudaMemcpy (data_out, scan_d, n_data_out * sizeof(int), cudaMemcpyDeviceToHost);
	//if (!input_on_device) mm->deviceFree<int>(data_d);
	//if (!output_on_device) mm->deviceFree<int>(scan_d);
}
