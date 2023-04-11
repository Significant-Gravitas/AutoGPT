#if defined(DETECT_FEATURES) && defined(__INTEL_COMPILER)
    /*
     * Unlike GCC and CLANG, Intel Compiler exposes all supported intrinsics,
     * whether or not the build options for those features are specified.
     * Therefore, we must test #definitions of CPU features when option native/host
     * is enabled via `--cpu-baseline` or through env var `CFLAGS` otherwise
     * the test will be broken and leads to enable all possible features.
     */
    #ifndef __AVX512VNNI__
        #error "HOST/ARCH doesn't support CascadeLake AVX512 features"
    #endif
#endif

#include <immintrin.h>

int main(int argc, char **argv)
{
    /* VNNI */
    __m512i a = _mm512_loadu_si512((const __m512i*)argv[argc-1]);
            a = _mm512_dpbusd_epi32(a, _mm512_setzero_si512(), a);
    return _mm_cvtsi128_si32(_mm512_castsi512_si128(a));
}
