#include <stdlib.h>
#include <string.h>

#include <cuda_runtime_api.h>

#include "compute_step.h"

compute_step_t new_compute_step (int N_in, int N_out, bool input_on_device, bool output_on_device) {
	compute_step_t cs;
	cs.n_data_in = N_in;
	cs.n_data_out = N_out;
	cs.input_on_device = input_on_device;
	cs.output_on_device = output_on_device;
	if (input_on_device) {
		cudaMalloc ((void**)&cs.data_in, N_in * sizeof(int));
	} else {
		cs.data_in = (int*)malloc(N_in * sizeof(int));
	}
	if (output_on_device) {
		cudaMalloc ((void**)&cs.data_out, N_out * sizeof(int));
	} else {
		cs.data_out = (int*)malloc(N_out * sizeof(int));
	}
	return cs;
}

compute_step_t cs_from_cs (compute_step_t cs_in, int N_out, bool output_on_device) {
	compute_step_t cs_out;
	cs_out.n_data_in = cs_in.n_data_out;
	cs_out.n_data_out = N_out;
	cs_out.input_on_device = cs_in.output_on_device;
	cs_out.output_on_device = output_on_device;
	cs_out.data_in = cs_in.data_out;
	if (output_on_device) {
		cudaMalloc((void**)&cs_out.n_data_out, N_out * sizeof(int));
	} else {
		cs_out.data_out = (int*)malloc(N_out * sizeof(int));
	}
	return cs_out;
}

void cs_pad (compute_step_t *cs, int pad_base) {
	int new_size = (cs->n_data_in / pad_base + 1) * pad_base;	
	if (cs->input_on_device) {
	   int *new_data;
	   cudaMalloc ((void**)&new_data, new_size * sizeof(int));
	   cudaMemset (new_data, 0, new_size * sizeof(int));
	   cudaMemcpy (new_data, cs->data_in, cs->n_data_in * sizeof(int), cudaMemcpyDeviceToDevice);
	   cudaFree(cs->data_in);
	   cs->data_in = new_data;
	} else {
	   cs->data_in = (int*)realloc (cs->data_in, new_size * sizeof(int));
	   memset (cs->data_in + cs->n_data_in, 0, (new_size - cs->n_data_in) * sizeof(int));
	}
	cs->n_data_in = new_size;
}

void cs_free (compute_step_t *cs) {
	if (cs->input_on_device) {
	   cudaFree(cs->data_in);
	} else {
	   free(cs->data_in);
        }	   
	if (cs->output_on_device) {
	   cudaFree(cs->data_out);
	} else {
	   free(cs->data_out);
	}
}
