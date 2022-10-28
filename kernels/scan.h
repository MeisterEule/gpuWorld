#ifndef SCAN_H
#define SCAN_H

#include "memoryManager.hpp"
#include "compute_step.hpp"

void scanArray (memoryManager *mm, int *data_in, int *data_out, int n_data_in, int n_data_out, bool input_on_device, bool output_on_device);

void scanArray (memoryManager *mm, ComputeStep<int,int> *cs);

#endif
