NVCC=nvcc
lib_LIBRARIES = libkernels.a

libkernels_a_SOURCES = random.cu \
		       count.cu \
		       reduction.cu \
		       grid_utils.cpp \
		       compute_step.cpp

libkernels_a_AR = $(NVCC) -lib -o

.cu.o : ; $(NVCC) $(NVCC_INCLUDE) $(NVCC_CXXFLAGS) -c -o $@ $<
