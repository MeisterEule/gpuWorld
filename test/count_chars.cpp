#include <stdio.h>
#include <stdlib.h>

#include "compute_step.hpp"
#include "memoryManager.hpp"
#include "count.h"

int main (int argc, char *argv[]) {
	memoryManager *mm = new memoryManager(false);
	char *sentence = "Berry the brown bulldog ate a big bouncy balloon.";

	ComputeStep<char,int> cs (mm, strlen(sentence) -1, 128, false, false, sentence);

	int *count_letters = countElementsInArray (mm, &cs);

	for (int i = 0; i < 128; i++) {
		if (count_letters[i] > 0) printf ("%c: %d\n", i, count_letters[i]);
	}
	return 0;
}
