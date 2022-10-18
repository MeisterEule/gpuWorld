#ifndef COMPUTE_STEP_H
#define COMPUTE_STEP_H

#include <stdbool.h>
#include <list>

class ComputeStep
{
   public:
      std::list<int> *n_data_in;
      std::list<int> *n_data_out;
      std::list<bool> *input_on_device;
      std::list<bool> *output_on_device;

      ComputeStep(int N_in, int N_out, bool i1, bool i2);
      ComputeStep(ComputeStep cs, int N_out, bool on_device);
};

class ComputeStepInt: public ComputeStep
{
   //using ComputeStep::ComputeStep;
   public:
	std::list<int*> *data_in;
	std::list<int*> *data_out;
        
        ComputeStepInt (int N_in, int N_out, bool i1, bool i2);
	ComputeStepInt (ComputeStepInt cs, int N_out, bool on_device);

	void SetInFirst (int *data);
 	void Pad (int padding_base);
	void Print ();
};


#endif
