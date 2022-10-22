#ifndef COUNT_H
#define COUNT_H

#include "memoryManager.hpp"
#include "compute_step.hpp"

void countElementsInArray (memoryManager *mm, ComputeStep<int,int> *cs);
int *countElementsInArray (memoryManager *mm, ComputeStep<char,int> *cs);


#endif
