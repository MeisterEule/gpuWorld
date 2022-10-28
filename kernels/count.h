#ifndef COUNT_H
#define COUNT_H

#include <stdbool.h>

#include "memoryManager.hpp"
#include "compute_step.hpp"

void countElementsInArray (memoryManager *mm, ComputeStep<int,int> *cs);

int *countElementsInArray (memoryManager *mm, ComputeStep<char,int> *cs);

unsigned long long countNonzeros (memoryManager *mm, float *data, LDIM n_data, bool input_on_device);
unsigned long long countNonzeros (memoryManager *mm, ComputeStep<float,int> *cs);


#endif
