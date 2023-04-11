#if defined(DETECT_FEATURES) && defined(__INTEL_COMPILER)
    /*
     * Unlike GCC and CLANG, Intel Compiler exposes all supported intrinsics,
     * whether or not the build options for those features are specified.
     * Therefore, we must test #definitions of CPU features when option native/host
     * is enabled via `--cpu-baseline` or through env var `CFLAGS` otherwise
     * the test will be broken and leads to enable all possible features.
     */
    #if !defined(__AVX5124FMAPS__) || !defined(__AVX5124VNNIW__) || !defined(__AVX512VPOPCNTDQ__)
        #error "HOST/ARCH doesn't support Knights Mill AVX512 features"
    #endif
#endif

#include <immintrin.h>

int main(int argc, char **argv)
{
    __m512i a = _mm512_loadu_si512((const __m512i*)argv[argc-1]);
    __m512 b = _mm512_loadu_ps((const __m512*)argv[argc-2]);

    /* 4FMAPS */
    b = _mm512_4fmadd_ps(b, b, b, b, b, NULL);
    /* 4VNNIW */
    a = _mm512_4dpwssd_epi32(a, a, a, a, a, NULL);
    /* VPOPCNTDQ */
    a = _mm512_popcnt_epi64(a);

    a = _mm512_add_epi32(a, _mm512_castps_si512(b));
    return _mm_cvtsi128_si32(_mm512_castsi512_si128(a));
}
