bin_PROGRAMS = count

count_SOURCES = count.cpp
count_CXXFLAGS = -I${srcdir}/../kernels
count_LDFLAGS = -L${top_builddir}/kernels/ -lkernels -L/usr/local/cuda-11.7/lib64 -lcudart
