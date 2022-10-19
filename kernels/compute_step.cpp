#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <list>

#include <cuda_runtime_api.h>

#include "compute_step.h"

ComputeStep::ComputeStep(int N_in, int N_out, bool i1, bool i2) {
   n_data_in = new std::list<int>;
   n_data_out = new std::list<int>;
   input_on_device = new std::list<bool>;
   output_on_device = new std::list<bool>;
   n_data_in->push_back(N_in);
   n_data_out->push_back(N_out);
   input_on_device->push_back(i1);
   output_on_device->push_back(i2);
}

ComputeStep::ComputeStep(ComputeStep cs, int N_out, bool on_device) {
   n_data_in = cs.n_data_out;
   input_on_device = cs.output_on_device;

   n_data_out = new std::list<int>;
   output_on_device = new std::list<bool>;
   n_data_out->push_back(N_out);
   output_on_device->push_back(on_device);
}

ComputeStepInt::ComputeStepInt(int N_in, int N_out, bool i1, bool i2): ComputeStep(N_in, N_out, i1, i2) {
	data_in = new std::list<int*>;
	data_out = new std::list<int*>;
	int *in, *out;
	if (i1) {
		cudaMalloc((void**)&in, N_in * sizeof(int));
	} else {
		in = (int*)malloc(N_in * sizeof(int));
	}

	if (i1) {
		cudaMalloc((void**)&out, N_out * sizeof(int));
	} else {
		out = (int*)malloc(N_out * sizeof(int));
	}

	data_in->push_back(in);
	data_out->push_back(out);
}

ComputeStepInt::ComputeStepInt(int N_in, int N_out, bool i1, bool i2, int *data): ComputeStep(N_in, N_out, i1, i2) {
	data_in = new std::list<int*>;
	data_out = new std::list<int*>;
	int *in, *out;
	if (i1) {
		cudaMalloc((void**)&in, N_in * sizeof(int));
		cudaMemcpy(in, data, N_in * sizeof(int), cudaMemcpyHostToDevice);
	} else {
		//in = (int*)malloc(N_in * sizeof(int));
		in = data;
	}

	if (i1) {
		cudaMalloc((void**)&out, N_out * sizeof(int));
	} else {
		out = (int*)malloc(N_out * sizeof(int));
	}

	data_in->push_back(in);
	data_out->push_back(out);
}

ComputeStepInt::ComputeStepInt(ComputeStepInt cs, int N_out, bool on_device): ComputeStep(cs, N_out, on_device) {
	data_in = cs.data_out;
	data_out = (std::list<int*>*)malloc(sizeof(std::list<int*>));
	int *out;
	if (on_device) {
		cudaMalloc((void**)&out, N_out * sizeof(int));
	} else {
		out = (int*)malloc(N_out * sizeof(int));
	}
	data_out->push_back(out);
}

void ComputeStepInt::SetInFirst (int *data) {
   int *this_data = data_in->front();
   this_data = data;
}

void ComputeStepInt::Pad (int padding_base) {
   std::list<int>::iterator it_n = n_data_in->begin();
   std::list<bool>::iterator it_od = input_on_device->begin();
   std::list<int*>::iterator it_data = data_in->begin(); 
   for (; it_n != n_data_in->end() && it_data != data_in->end() && it_od != input_on_device->end();
          ++it_n, ++it_data, ++it_od) {
      int new_size = (*it_n / padding_base + 1) * padding_base;
      if (*it_od) {
         int *new_data;
	 cudaMalloc ((void**)&new_data, new_size * sizeof(int));
         cudaMemset (new_data, 0, new_size * sizeof(int));
         cudaMemcpy (new_data, *it_data, *it_n * sizeof(int), cudaMemcpyDeviceToDevice); 
         cudaFree (*it_data);
	 *it_data = new_data;
      } else {
         *it_data = (int*)realloc (*it_data, new_size * sizeof(int));
         memset (*it_data + *it_n, 0, (new_size - *it_n) * sizeof(int));
      }
      *it_n = new_size;
   }
}

void ComputeStepInt::PrintIn () {
   printf ("In States: %d\n", n_data_in->size());  
   std::list<int>::iterator it_n = n_data_in->begin();
   std::list<int*>::iterator it_data = data_in->begin();

   int n_states = 0;
   for (; it_n != n_data_in->end() && it_data != data_in->end(); ++it_n, ++it_data) {
      printf ("%d: ", n_states++);
      int *tmp = *it_data;
      for (int i = 0; i < *it_n; i++) {
	      printf ("%d ", tmp[i]);
      }
      printf ("\n");
   }
}

void ComputeStepInt::PrintOut () {
   printf ("Out States: %d\n", n_data_in->size());  
   std::list<int>::iterator it_n = n_data_out->begin();
   std::list<int*>::iterator it_data = data_out->begin();

   int n_states = 0;
   for (; it_n != n_data_out->end() && it_data != data_out->end(); ++it_n, ++it_data) {
      printf ("%d: ", n_states++);
      int *tmp = *it_data;
      for (int i = 0; i < *it_n; i++) {
	      printf ("%d ", tmp[i]);
      }
      printf ("\n");
   }
}
