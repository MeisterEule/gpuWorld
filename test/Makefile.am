bin_PROGRAMS = reduce \
	       count \
	       count_chars

reduce_SOURCES = reduce.cpp
reduce_CXXFLAGS = -I${srcdir}/../kernels -I${srcdir}/../classes
reduce_LDFLAGS = -L${top_builddir}/kernels/ -lkernels

count_SOURCES = count.cpp
count_CXXFLAGS = -I${srcdir}/../kernels -I${srcdir}/../classes
count_LDFLAGS = -L${top_builddir}/kernels/ -lkernels

count_chars_SOURCES = count.cpp
count_chars_CXXFLAGS = -I${srcdir}/../kernels -I${srcdir}/../classes
count_chars_LDFLAGS = -L${top_builddir}/kernels/ -lkernels
