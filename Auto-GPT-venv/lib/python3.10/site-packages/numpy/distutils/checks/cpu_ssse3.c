#if defined(DETECT_FEATURES) && defined(__INTEL_COMPILER)
    /*
     * Unlike GCC and CLANG, Intel Compiler exposes all supported intrinsics,
     * whether or not the build options for those features are specified.
     * Therefore, we must test #definitions of CPU features when option native/host
     * is enabled via `--cpu-baseline` or through env var `CFLAGS` otherwise
     * the test will be broken and leads to enable all possible features.
     */
    #ifndef __SSSE3__
        #error "HOST/ARCH doesn't support SSSE3"
    #endif
#endif

#include <tmmintrin.h>

int main(void)
{
    __m128i a = _mm_hadd_epi16(_mm_setzero_si128(), _mm_setzero_si128());
    return (int)_mm_cvtsi128_si32(a);
}
