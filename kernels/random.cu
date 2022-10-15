#include <stdint.h>

#include <curand.h>
#include <curand_kernel.h>

#include "grid_utils.h"

__global__ void setup_curand_kernel (curandState *state, uint64_t seed) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(seed, tid, 0, &state[tid]);
}

__global__ void fill_array_kernel (int *data, int N, int min, int max, curandState *globalState) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= N) return;
	curandState localState = globalState[tid];

 	data[tid] = min + curand_uniform(&localState) * (min - max);
}

static curandState *deviceCurandStates;
static int curand_threads;
static int curand_blocks;


void initRNG (uint64_t seed, int N_numbers) {
   getGridDimension1D (N_numbers, &curand_blocks, &curand_threads);
   cudaMalloc(&deviceCurandStates, curand_threads * curand_blocks * sizeof(curandState));
   setup_curand_kernel<<<curand_blocks,curand_threads>>>(deviceCurandStates, seed);
}

int *generateRandomArrayInt (int N, int min, int max) {
   int *rng_data_h = (int*)malloc(N * sizeof(int));
   int *rng_data_d;
   cudaMalloc((void**)&rng_data_d, N * sizeof(int));
   fill_array_kernel<<<curand_blocks,curand_threads>>>(rng_data_d, N, min, max, deviceCurandStates);
   cudaMemcpy(rng_data_h, rng_data_d, N * sizeof(int), cudaMemcpyDeviceToHost);
   cudaFree(rng_data_d);
   return rng_data_h;
}
