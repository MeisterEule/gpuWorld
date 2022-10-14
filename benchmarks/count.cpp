#include <stdio.h>
#include <stdlib.h>

#include "count.h"

int main (int argc, char *argv[]) {
	int N = 10;
	int *data = (int*)malloc(N * sizeof(int));

	for (int i = 0; i < N; i++) {
		data[i] = i / 3;
	}
    	
	int *count = countElementsInArray (data, N);

   	for (int i = 0; i < N; i++) {
		printf ("%d: %d\n", data[i], count[i]);
	}	
}
