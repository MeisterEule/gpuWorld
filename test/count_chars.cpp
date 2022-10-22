#include <stdio.h>
#include <stdlib.h>

#include "compute_step.hpp"
#include "memoryManager.hpp"
#include "count.h"

int main (int argc, char *argv[]) {
	char *sentence = "berrythebrownbulldog";

	ComputeStep<char> cs (mm, strlen(sentence) -1, 26, false, false, sentence);
}
