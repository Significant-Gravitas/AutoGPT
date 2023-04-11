#if defined(DETECT_FEATURES) && defined(__INTEL_COMPILER)
    /*
     * Unlike GCC and CLANG, Intel Compiler exposes all supported intrinsics,
     * whether or not the build options for those features are specified.
     * Therefore, we must test #definitions of CPU features when option native/host
     * is enabled via `--cpu-baseline` or through env var `CFLAGS` otherwise
     * the test will be broken and leads to enable all possible features.
     */
    #ifndef __AVX2__
        #error "HOST/ARCH doesn't support AVX2"
    #endif
#endif

#include <immintrin.h>

int main(int argc, char **argv)
{
    __m256i a = _mm256_abs_epi16(_mm256_loadu_si256((const __m256i*)argv[argc-1]));
    return _mm_cvtsi128_si32(_mm256_castsi256_si128(a));
}
