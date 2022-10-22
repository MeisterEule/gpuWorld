#ifndef MEMORYMANAGER_HPP
#define MEMORYMANAGER_HPP

#include <stdio.h>
#include <iostream>
#include <stdbool.h>

#include <cuda_runtime_api.h>

class memoryManager {
	public:
		size_t total_free_host_memory;
		size_t used_host_memory;
		size_t total_device_memory;
		size_t used_device_memory;		
		bool verbose;

		memoryManager (bool verb);
		template<class T> int deviceAllocate(T *&var, size_t count);
		// Implementing deviceFree requires us to better keep track in detail
		// which fields are allocated. Or we just check the device memory after freeing (which takes much more time).
		void status();
};

// Leaving out the inline leads to multiple definition errors when linking the benchmarks.
// Strangely, this does not happen with the compute steps....
inline memoryManager::memoryManager (bool verb) {
	cudaDeviceProp props;
	cudaGetDeviceProperties (&props, 0);
	total_device_memory = props.totalGlobalMem;
	used_device_memory = 0;
	// not implemented
	total_free_host_memory = 0;
	used_host_memory = 0;
	verbose = verb;
}

template<typename T> int memoryManager::deviceAllocate (T *&var, size_t count) {
	size_t required_memory = count * sizeof(T);
	if (verbose) printf ("Allocating %lf GiB\n", (double)count * sizeof(T) / 1024 / 1024 / 1024);
	if (used_device_memory + required_memory > total_device_memory) {
		printf ("Not enough memory:\n");
		status();
		std::abort();
	}	
	cudaError_t ce = cudaMalloc((void**)&var, count * sizeof(T));
	used_device_memory += required_memory;
	return 0;
}

inline void memoryManager::status (){
	printf ("Device memory: %ld / %ld\n", used_device_memory, total_device_memory);
}

#endif
