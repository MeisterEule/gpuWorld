#ifndef ARRAY_GENERATORS_HPP
#define ARRAY_GENERATORS_HPP

#include <stdlib.h>

#include "memoryManager.hpp"
#include "random.h"

template<typename T> T *generateArrayOfOnesCPU (int N) {
	T *x = (T*)malloc(N * sizeof(T));
	for (int i = 0; i < N; i++) {
		x[i] = (T)1;
	}
	return x;
}

//inline int *generateRandomNumbersGPU (memoryManager *mm, int N, int min, int max) {
//   return generateRandomArrayInt (mm, N, min, max);
//}

#endif
