bin_PROGRAMS = reduce \
	       count \
	       count_chars \
	       random

reduce_SOURCES = reduce.cpp
reduce_CXXFLAGS = -I${srcdir}/../kernels -I${srcdir}/../classes
reduce_LDFLAGS = -L${top_builddir}/kernels/ -lkernels

count_SOURCES = count.cpp
count_CXXFLAGS = -I${srcdir}/../kernels -I${srcdir}/../classes
count_LDFLAGS = -L${top_builddir}/kernels/ -lkernels

count_chars_SOURCES = count_chars.cpp
count_chars_CXXFLAGS = -I${srcdir}/../kernels -I${srcdir}/../classes
count_chars_LDFLAGS = -L${top_builddir}/kernels/ -lkernels

random_SOURCES = random.cpp
random_CXXFLAGS = -I${srcdir}/../kernels -I${srcdir}/../classes
random_LDFLAGS = -L${top_builddir}/kernels/ -lkernels

