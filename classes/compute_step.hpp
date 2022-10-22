#ifndef COMPUTE_STEP_HPP
#define COMPUTE_STEP_HPP

#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <math.h>
#include <list>

#include <cuda_runtime_api.h>

#include "memoryManager.hpp"

template<typename T, typename U> class ComputeStep 
{
	public:
	   memoryManager *mm;
	   std::list<int> *n_data_in;
	   std::list<int> *n_data_out;
	   std::list<bool> *input_on_device;
	   std::list<bool> *output_on_device;
	   std::list<T*> *data_in;
	   std::list<U*> *data_out;

	   ComputeStep (memoryManager *mm, int N_in, int N_out, bool i1, bool i2);
	   ComputeStep (memoryManager *mm, int N_in, int N_out, bool i1, bool i2, T* data);
	   ComputeStep (memoryManager *mm, ComputeStep<T,U> cs, int N_out, bool on_device);

	   void SetInFirst (T* data);
	   void Pad (int padding_base);
	   void PrintIn ();
	   void PrintOut ();
};

template<typename T, typename U> ComputeStep<T,U>::ComputeStep(memoryManager *mm, int N_in, int N_out, bool i1, bool i2) {
	mm = mm;
     	n_data_in = new std::list<int>;
        n_data_out = new std::list<int>;
        input_on_device = new std::list<bool>;
        output_on_device = new std::list<bool>;
        n_data_in->push_back(N_in);
        n_data_out->push_back(N_out);
        input_on_device->push_back(i1);
        output_on_device->push_back(i2);

	data_in = new std::list<T*>;
	data_out = new std::list<U*>;
	T *in;
	U *out;
	if (i1) {
		mm->deviceAllocate<T> (in, N_in);
	} else {
		in = (T*)malloc(N_in * sizeof(T));
	}

	if (i2) {
		mm->deviceAllocate<U> (out, N_out);
	} else {
		out = (U*)malloc(N_out * sizeof(U));
	}

	data_in->push_back(in);
	data_out->push_back(out);
}

template<typename T, typename U> ComputeStep<T,U>::ComputeStep(memoryManager *mm, int N_in, int N_out, bool i1, bool i2, T *data) {
	mm = mm;
	n_data_in = new std::list<int>;
        n_data_out = new std::list<int>;
        input_on_device = new std::list<bool>;
        output_on_device = new std::list<bool>;
        n_data_in->push_back(N_in);
        n_data_out->push_back(N_out);
        input_on_device->push_back(i1);
        output_on_device->push_back(i2);

	data_in = new std::list<T*>;
	data_out = new std::list<U*>;
	T *in;
	U *out;
	if (i1) {
		mm->deviceAllocate<T> (in, N_in);
		cudaMemcpy(in, data, N_in * sizeof(T), cudaMemcpyHostToDevice);
	} else {
		in = data;
	}

	if (i2) {
		mm->deviceAllocate<U> (out, N_out);
	} else {
		out = (U*)malloc(N_out * sizeof(U));
	}

	data_in->push_back(in);
	data_out->push_back(out);
}

template<typename T, typename U> ComputeStep<T,U>::ComputeStep(memoryManager *mm, ComputeStep<T,U> cs, int N_out, bool on_device) {
	mm = mm;
     	n_data_in = cs.n_data_out;
   	input_on_device = cs.output_on_device;
   	n_data_out = new std::list<int>;
   	output_on_device = new std::list<bool>;
   	n_data_out->push_back(N_out);
   	output_on_device->push_back(on_device);

	data_in = cs.data_out;
	data_out = new std::list<U*>;
	U *out;
	if (on_device) {
		mm->deviceAllocate<U> (out, N_out);
	} else {
		out = (U*)malloc(N_out * sizeof(U));
	}
	data_out->push_back(out);
}

template<typename T, typename U> void ComputeStep<T,U>::SetInFirst (T *data) {
   int *this_data = data_in->front();
   this_data = data;
}

template<typename T, typename U> void ComputeStep<T,U>::Pad (int padding_base) {
   std::list<int>::iterator it_n = n_data_in->begin();
   std::list<bool>::iterator it_od = input_on_device->begin();
   typename std::list<T*>::iterator it_data = data_in->begin();  // I have no idea why it requires 'typename'
   for (; it_n != n_data_in->end() && it_data != data_in->end() && it_od != input_on_device->end();
          ++it_n, ++it_data, ++it_od) {
      int new_size = (*it_n / padding_base + 1) * padding_base;
      if (*it_od) {
         T *new_data;
         cudaMalloc ((void**)&new_data, new_size * sizeof(T));
         cudaMemset (new_data, 0, new_size * sizeof(T));
         cudaMemcpy (new_data, *it_data, *it_n * sizeof(T), cudaMemcpyDeviceToDevice); 
         cudaFree (*it_data);
         *it_data = new_data;
      } else {
         *it_data = (T*)realloc (*it_data, new_size * sizeof(T));
         memset (*it_data + *it_n, 0, (new_size - *it_n) * sizeof(T));
      }
      *it_n = new_size;
   }
}

template<typename T, typename U> void ComputeStep<T,U>::PrintIn () {
   std::cout << "In States: " << n_data_in->size() << std::endl;
   std::list<int>::iterator it_n = n_data_in->begin();
   typename std::list<T*>::iterator it_data = data_in->begin();

   int n_states = 0;
   for (; it_n != n_data_in->end() && it_data != data_in->end(); ++it_n, ++it_data) {
      std::cout << n_states++;
      T *tmp = *it_data;
      for (int i = 0; i < *it_n; i++) {
	      std::cout << tmp[i] << " ";
      }
      std::cout << std::endl;
   }
}

template<typename T, typename U> void ComputeStep<T,U>::PrintOut () {
   std::cout << "Out States: " << n_data_out->size() << std::endl;
   std::list<int>::iterator it_n = n_data_out->begin();
   typename std::list<U*>::iterator it_data = data_out->begin();

   int n_states = 0;
   for (; it_n != n_data_out->end() && it_data != data_out->end(); ++it_n, ++it_data) {
      std::cout << n_states++;
      U *tmp = *it_data;
      for (int i = 0; i < *it_n; i++) {
	      std::cout << tmp[i] << " ";
      }
      std::cout << std::endl;
   }
}

#endif
