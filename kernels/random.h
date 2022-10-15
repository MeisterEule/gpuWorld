#ifndef RANDOM_H
#define RANDOM_H

#include <stdint.h>

#define DEFAULT_SEED 12345

void initRNG (uint64_t seed, int n_Numbers);
int *generateRandomArrayInt (int N, int min, int max);

#endif
