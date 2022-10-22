NVCC=nvcc
NVCC_INCLUDE = -I${srcdir}/../classes
lib_LIBRARIES = libkernels.a

libkernels_a_SOURCES = random.cu \
		       count.cu \
		       reduction.cu \
		       grid_utils.cpp


libkernels_a_AR = $(NVCC) -lib -o

.cu.o : ; $(NVCC) $(NVCC_INCLUDE) $(NVCC_CXXFLAGS) -c -o $@ $<
