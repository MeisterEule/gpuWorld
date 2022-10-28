#ifndef RANDOM_H
#define RANDOM_H

#include <stdint.h>

#include <curand_kernel.h>

#include "types.h"
#include "memoryManager.hpp"

#define DEFAULT_SEED 12345

class cudaRNG {
	public:
		curandState *deviceCurandStates;
		size_t reserved_bytes;
		int gen_stride;
		int n_threads;
		int n_blocks;
		uint64_t seed;

		cudaRNG (size_t bytes, uint64_t init_seed);
		void initRNG (memoryManager *mm, int N_numbers);
		int *generate (memoryManager *mm, int N_numbers, int min, int max);
		float *generateRandomMatrix (memoryManager *mm, LDIM N_numbers, float nonzero_ratio);
		void freeRNG (memoryManager *mm);
		void printStatus ();
};

#endif
