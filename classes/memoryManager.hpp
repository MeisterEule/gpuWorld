#ifndef MEMORYMANAGER_HPP
#define MEMORYMANAGER_HPP

#include <stdio.h>
#include <iostream>
#include <stdbool.h>
#include <map>

#include <cuda_runtime_api.h>

#include "types.h"

typedef struct {
	char *name;
	LDIM this_memory;
} mem_registry_t;


class memoryManager {
	public:
		std::map<uint64_t,mem_registry_t> active_fields;
		LDIM total_free_host_memory;
		LDIM used_host_memory;
		LDIM total_device_memory;
		LDIM used_device_memory;		
		bool verbose;

		memoryManager (bool verb);
		template<class T> cudaError_t deviceAllocate(T *&var, LDIM count, char *name = "unnamed");
		template<class T> void deviceFree (T *&var);
		template<class T> bool isPossible (LDIM n);
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

template<class T> cudaError_t memoryManager::deviceAllocate (T *&var, LDIM count, char *name) {
	LDIM required_memory = count * sizeof(T);
	mem_registry_t m;
	m.name = name;
	m.this_memory = required_memory;
	if (verbose) printf ("Allocating %lf GiB (%s @ 0x%lx)\n", (double)m.this_memory / 1024 / 1024 / 1024, m.name, (uint64_t)&var);
	active_fields[(uint64_t)&var] = m;
	if (used_device_memory + required_memory > total_device_memory) {
		printf ("Not enough memory:\n");
		status();
		std::abort();
	}	
	cudaError_t ce = cudaMalloc((void**)&var, count * sizeof(T));
	used_device_memory += required_memory;
	return ce;
}

template<class T> void memoryManager::deviceFree (T *&var) {
	uint64_t var_addr = (uint64_t)&var;
	if (verbose) printf ("Freeing 0x%lx (%s)\n", var_addr, active_fields[var_addr].name);
	cudaFree(var);
	used_device_memory -= active_fields[var_addr].this_memory;
	active_fields.erase(var_addr);
}

inline void memoryManager::status (){
	printf ("Device memory: %lf GiB / %lf GiB\n", (double)used_device_memory / 1024 / 1024 / 1024, (double)total_device_memory / 1024 / 1024 / 1024);
	std::map<uint64_t,mem_registry_t>::iterator it = active_fields.begin();
	for (; it != active_fields.end(); ++it) {
		printf ("%s (@0x%lx): %lld B\n", it->second.name, it->first, it->second.this_memory);
	}
}

template<class T> bool memoryManager::isPossible (LDIM n) {
	return n * sizeof(T) < total_device_memory;
}

#endif
