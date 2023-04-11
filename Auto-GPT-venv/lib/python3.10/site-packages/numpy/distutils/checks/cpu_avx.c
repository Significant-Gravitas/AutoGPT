#if defined(DETECT_FEATURES) && defined(__INTEL_COMPILER)
    /*
     * Unlike GCC and CLANG, Intel Compiler exposes all supported intrinsics,
     * whether or not the build options for those features are specified.
     * Therefore, we must test #definitions of CPU features when option native/host
     * is enabled via `--cpu-baseline` or through env var `CFLAGS` otherwise
     * the test will be broken and leads to enable all possible features.
     */
    #ifndef __AVX__
        #error "HOST/ARCH doesn't support AVX"
    #endif
#endif

#include <immintrin.h>

int main(int argc, char **argv)
{
    __m256 a = _mm256_add_ps(_mm256_loadu_ps((const float*)argv[argc-1]), _mm256_loadu_ps((const float*)argv[1]));
    return (int)_mm_cvtss_f32(_mm256_castps256_ps128(a));
}
