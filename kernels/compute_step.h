#ifndef COMPUTE_STEP_H
#define COMPUTE_STEP_H

#include <stdbool.h>

typedef struct {
	int n_data_in;
	int n_data_out;
	int *data_in;
	int *data_out;
 	bool input_on_device;
	bool output_on_device;
} compute_step_t;

compute_step_t new_compute_step (int N_in, int N_out, bool input_on_device, bool output_on_device);
compute_step_t cs_from_cs (compute_step_t cs_in, int N_out, bool output_on_device);
void cs_free (compute_step_t *cs);
void cs_pad (compute_step_t *cs, int pad_base);

#endif
