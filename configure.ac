AC_INIT([gpuWorld], [0.0.0])
: ${CFLAGS="-O2"}
: ${FCFLAGS="-O2"}

AC_CONFIG_HEADERS([config.h])
AC_CONFIG_MACRO_DIRS([m4])
AM_INIT_AUTOMAKE([-Wall -Werror foreign no-dependencies subdir-objects])

AC_PROG_CXX
AC_ENABLE_SHARED
AM_PROG_AR
LT_INIT
AX_CHECK_CXX_STDFLAG

AC_CONFIG_FILES([Makefile])
AC_CONFIG_FILES([kernels/Makefile])
AC_CONFIG_FILES([benchmarks/Makefile])

AC_OUTPUT
