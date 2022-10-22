#ifndef COMPUTE_STEP_HPP
#define COMPUTE_STEP_HPP

#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <math.h>
#include <list>

#include <cuda_runtime_api.h>

template<typename T> void cudaMallocT(T *&d_var, std::size_t count) {
	cudaError_t ce = cudaMalloc((void**)&d_var, count * sizeof(T));
	std::cout << cudaGetErrorString(ce) << std::endl;
}

template<typename T> class ComputeStep 
{
	public:
	   std::list<int> *n_data_in;
	   std::list<int> *n_data_out;
	   std::list<bool> *input_on_device;
	   std::list<bool> *output_on_device;
	   std::list<T*> *data_in;
	   std::list<T*> *data_out;

	   ComputeStep (int N_in, int N_out, bool i1, bool i2);
	   ComputeStep (int N_in, int N_out, bool i1, bool i2, T* data);
	   ComputeStep (ComputeStep<T> cs, int N_out, bool on_device);

	   void SetInFirst (T* data);
	   void Pad (int padding_base);
	   void PrintIn ();
	   void PrintOut ();
};

template<typename T> ComputeStep<T>::ComputeStep(int N_in, int N_out, bool i1, bool i2) {
     	n_data_in = new std::list<int>;
        n_data_out = new std::list<int>;
        input_on_device = new std::list<bool>;
        output_on_device = new std::list<bool>;
        n_data_in->push_back(N_in);
        n_data_out->push_back(N_out);
        input_on_device->push_back(i1);
        output_on_device->push_back(i2);

	data_in = new std::list<T*>;
	data_out = new std::list<T*>;
	T *in, *out;
	if (i1) {
		cudaMallocT<T>(in, N_in);
	} else {
		in = (T*)malloc(N_in * sizeof(T));
	}

	if (i1) {
		cudaMallocT<T>(out, N_out);
	} else {
		out = (T*)malloc(N_out * sizeof(T));
	}

	data_in->push_back(in);
	data_out->push_back(out);
}

template<typename T> ComputeStep<T>::ComputeStep(int N_in, int N_out, bool i1, bool i2, T *data) {
	printf ("HUHU\n");
	n_data_in = new std::list<int>;
        n_data_out = new std::list<int>;
        input_on_device = new std::list<bool>;
        output_on_device = new std::list<bool>;
        n_data_in->push_back(N_in);
        n_data_out->push_back(N_out);
        input_on_device->push_back(i1);
        output_on_device->push_back(i2);

	data_in = new std::list<T*>;
	data_out = new std::list<T*>;
	T *in, *out;
	if (i1) {
		cudaMallocT<T>(in, N_in);
		cudaMemcpy(in, data, N_in * sizeof(T), cudaMemcpyHostToDevice);
	} else {
		in = data;
	}

	if (i1) {
		cudaMallocT<T>(out, N_out);
	} else {
		out = (T*)malloc(N_out * sizeof(T));
	}

	data_in->push_back(in);
	data_out->push_back(out);
}

template<typename T> ComputeStep<T>::ComputeStep(ComputeStep<T> cs, int N_out, bool on_device) {
     	n_data_in = cs.n_data_out;
   	input_on_device = cs.output_on_device;
   	n_data_out = new std::list<int>;
   	output_on_device = new std::list<bool>;
   	n_data_out->push_back(N_out);
   	output_on_device->push_back(on_device);

	data_in = cs.data_out;
	data_out = new std::list<T*>;
	T *out;
	if (on_device) {
		cudaMallocT<T>(out, N_out);
	} else {
		out = (T*)malloc(N_out * sizeof(T));
	}
	data_out->push_back(out);
}

template<typename T> void ComputeStep<T>::SetInFirst (T *data) {
   int *this_data = data_in->front();
   this_data = data;
}

template<typename T> void ComputeStep<T>::Pad (int padding_base) {
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

template<typename T> void ComputeStep<T>::PrintIn () {
   //printf ("In States: %d\n", n_data_in->size());  
   std::cout << "In States: " << n_data_in->size() << std::endl;
   std::list<int>::iterator it_n = n_data_in->begin();
   typename std::list<T*>::iterator it_data = data_in->begin();

   int n_states = 0;
   for (; it_n != n_data_in->end() && it_data != data_in->end(); ++it_n, ++it_data) {
      //printf ("%d: ", n_states++);
      std::cout << n_states++;
      T *tmp = *it_data;
      for (int i = 0; i < *it_n; i++) {
	      //printf ("%d ", tmp[i]);
	      std::cout << tmp[i] << " ";
      }
      //printf ("\n");
      std::cout << std::endl;
   }
}

template<typename T> void ComputeStep<T>::PrintOut () {
   //printf ("Out States: %d\n", n_data_in->size());  
   std::cout << "Out States: " << n_data_out->size() << std::endl;
   std::list<int>::iterator it_n = n_data_out->begin();
   typename std::list<T*>::iterator it_data = data_out->begin();

   int n_states = 0;
   for (; it_n != n_data_out->end() && it_data != data_out->end(); ++it_n, ++it_data) {
      //printf ("%d: ", n_states++);
      std::cout << n_states++;
      T *tmp = *it_data;
      for (int i = 0; i < *it_n; i++) {
	      //printf ("%d ", tmp[i]);
	      std::cout << tmp[i] << " ";
      }
      //printf ("\n");
      std::cout << std::endl;
   }
}

#endif
