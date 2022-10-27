#ifndef COUNT_H
#define COUNT_H

#include <stdbool.h>

#include "memoryManager.hpp"
#include "compute_step.hpp"

void countElementsInArray (memoryManager *mm, ComputeStep<int,int> *cs);

int *countElementsInArray (memoryManager *mm, ComputeStep<char,int> *cs);

//template <typename T> int countNonzeros (memoryManager *mm, T *data, int n_data, bool input_on_device);
//template <typename T> int countNonzeros (memoryManager *mm, ComputeStep<T,T> *cs);
int countNonzeros (memoryManager *mm, float *data, int n_data, bool input_on_device);
int countNonzeros (memoryManager *mm, ComputeStep<float,int> *cs);


#endif
