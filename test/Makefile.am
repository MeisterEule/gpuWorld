bin_PROGRAMS = reduce \
	       count

reduce_SOURCES = reduce.cpp
reduce_CXXFLAGS = -I${srcdir}/../kernels -I${srcdir}/../classes
reduce_LDFLAGS = -L${top_builddir}/kernels/ -lkernels

count_SOURCES = count.cpp
count_CXXFLAGS = -I${srcdir}/../kernels -I${srcdir}/../classes
count_LDFLAGS = -L${top_builddir}/kernels/ -lkernels
