#ifndef NUMPY_CORE_INCLUDE_NUMPY_NPY_NUMPYCONFIG_H_
#define NUMPY_CORE_INCLUDE_NUMPY_NPY_NUMPYCONFIG_H_

#include "_numpyconfig.h"

/*
 * On Mac OS X, because there is only one configuration stage for all the archs
 * in universal builds, any macro which depends on the arch needs to be
 * hardcoded.
 *
 * Note that distutils/pip will attempt a universal2 build when Python itself
 * is built as universal2, hence this hardcoding is needed even if we do not
 * support universal2 wheels anymore (see gh-22796).
 * This code block can be removed after we have dropped the setup.py based
 * build completely.
 */
#ifdef __APPLE__
    #undef NPY_SIZEOF_LONG
    #undef NPY_SIZEOF_PY_INTPTR_T

    #ifdef __LP64__
        #define NPY_SIZEOF_LONG         8
        #define NPY_SIZEOF_PY_INTPTR_T  8
    #else
        #define NPY_SIZEOF_LONG         4
        #define NPY_SIZEOF_PY_INTPTR_T  4
    #endif

    #undef NPY_SIZEOF_LONGDOUBLE
    #undef NPY_SIZEOF_COMPLEX_LONGDOUBLE
    #ifdef HAVE_LDOUBLE_IEEE_DOUBLE_LE
      #undef HAVE_LDOUBLE_IEEE_DOUBLE_LE
    #endif
    #ifdef HAVE_LDOUBLE_INTEL_EXTENDED_16_BYTES_LE
      #undef HAVE_LDOUBLE_INTEL_EXTENDED_16_BYTES_LE
    #endif

    #if defined(__arm64__)
        #define NPY_SIZEOF_LONGDOUBLE         8
        #define NPY_SIZEOF_COMPLEX_LONGDOUBLE 16
        #define HAVE_LDOUBLE_IEEE_DOUBLE_LE 1
    #elif defined(__x86_64)
        #define NPY_SIZEOF_LONGDOUBLE         16
        #define NPY_SIZEOF_COMPLEX_LONGDOUBLE 32
        #define HAVE_LDOUBLE_INTEL_EXTENDED_16_BYTES_LE 1
    #elif defined (__i386)
        #define NPY_SIZEOF_LONGDOUBLE         12
        #define NPY_SIZEOF_COMPLEX_LONGDOUBLE 24
    #elif defined(__ppc__) || defined (__ppc64__)
        #define NPY_SIZEOF_LONGDOUBLE         16
        #define NPY_SIZEOF_COMPLEX_LONGDOUBLE 32
    #else
        #error "unknown architecture"
    #endif
#endif


/**
 * To help with the NPY_NO_DEPRECATED_API macro, we include API version
 * numbers for specific versions of NumPy. To exclude all API that was
 * deprecated as of 1.7, add the following before #including any NumPy
 * headers:
 *   #define NPY_NO_DEPRECATED_API  NPY_1_7_API_VERSION
 */
#define NPY_1_7_API_VERSION 0x00000007
#define NPY_1_8_API_VERSION 0x00000008
#define NPY_1_9_API_VERSION 0x00000008
#define NPY_1_10_API_VERSION 0x00000008
#define NPY_1_11_API_VERSION 0x00000008
#define NPY_1_12_API_VERSION 0x00000008
#define NPY_1_13_API_VERSION 0x00000008
#define NPY_1_14_API_VERSION 0x00000008
#define NPY_1_15_API_VERSION 0x00000008
#define NPY_1_16_API_VERSION 0x00000008
#define NPY_1_17_API_VERSION 0x00000008
#define NPY_1_18_API_VERSION 0x00000008
#define NPY_1_19_API_VERSION 0x00000008
#define NPY_1_20_API_VERSION 0x0000000e
#define NPY_1_21_API_VERSION 0x0000000e
#define NPY_1_22_API_VERSION 0x0000000f
#define NPY_1_23_API_VERSION 0x00000010
#define NPY_1_24_API_VERSION 0x00000010

#endif  /* NUMPY_CORE_INCLUDE_NUMPY_NPY_NUMPYCONFIG_H_ */
