#include <stdint.h>
#include <stdio.h>

#include <curand.h>
#include <curand_kernel.h>

#include "memoryManager.hpp"
#include "grid_utils.h"
#include "random.h"

__global__ void setup_curand_kernel (curandState *state, uint64_t seed) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(seed, tid, 0, &state[tid]);
}

__global__ void fill_array_kernel (int *data, int N, int min, int max, int stride, curandState *globalState) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= N) return;
	curandState localState = globalState[tid];

	tid *= stride;
	for (int i = 0; i < stride && tid + i < N; i++) {
	   double f = (double)curand_uniform(&localState);
 	   data[tid + i] = min + (int)(f * (max - min));
	}
}

cudaRNG::cudaRNG (size_t bytes, uint64_t init_seed) {
	reserved_bytes = bytes;
	seed = init_seed;
	gen_stride = 0;
	n_threads = 0;
	n_blocks = 0;
}

void cudaRNG::initRNG (memoryManager *mm, int N_numbers) {
	int n_curand_states = reserved_bytes / sizeof(curandState) + 1;
	getGridDimension1D (n_curand_states, &n_blocks, &n_threads);
	gen_stride = N_numbers / n_curand_states;
	if (gen_stride == 0) gen_stride = 1;
        mm->deviceAllocate<curandState>(deviceCurandStates, n_threads * n_blocks, "curandStates");
        setup_curand_kernel<<<n_blocks,n_threads>>>(deviceCurandStates, seed);
}

int *cudaRNG::generate (memoryManager *mm, int N_numbers, int min, int max) {
   int *rng_data_h = (int*)malloc(N_numbers * sizeof(int));
   int *rng_data_d;
   mm->deviceAllocate<int> (rng_data_d, N_numbers, "devRandomNumbers");

   fill_array_kernel<<<n_blocks,n_threads>>>(rng_data_d, N_numbers, min, max, gen_stride, deviceCurandStates);
   cudaMemcpy(rng_data_h, rng_data_d, N_numbers * sizeof(int), cudaMemcpyDeviceToHost);
   mm->deviceFree (rng_data_d, (uint64_t)&rng_data_d);
   return rng_data_h;
}

void cudaRNG::printStatus () {
	printf ("cudaRNG: \n");
	printf ("  reserved bytes: %d\n", reserved_bytes);
	printf ("  seed: %lld\n", seed);
	printf ("  n_blocks: %d\n", n_blocks);
	printf ("  n_threads: %d\n", n_threads);
	printf ("  stride: %d\n", gen_stride);
}
