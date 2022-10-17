#include <stdlib.h>

#include "count.h"
#include "random.h"

int main (int argc, char *argv[]) {
	int N = 10000;
	initRNG (DEFAULT_SEED, N);
	int *random_numbers = generateRandomArrayInt (N, 0, 100);

	free(random_numbers);	
}