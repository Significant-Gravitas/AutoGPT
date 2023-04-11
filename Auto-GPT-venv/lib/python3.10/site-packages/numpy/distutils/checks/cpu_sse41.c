#if defined(DETECT_FEATURES) && defined(__INTEL_COMPILER)
    /*
     * Unlike GCC and CLANG, Intel Compiler exposes all supported intrinsics,
     * whether or not the build options for those features are specified.
     * Therefore, we must test #definitions of CPU features when option native/host
     * is enabled via `--cpu-baseline` or through env var `CFLAGS` otherwise
     * the test will be broken and leads to enable all possible features.
     */
    #ifndef __SSE4_1__
        #error "HOST/ARCH doesn't support SSE41"
    #endif
#endif

#include <smmintrin.h>

int main(void)
{
    __m128 a = _mm_floor_ps(_mm_setzero_ps());
    return (int)_mm_cvtss_f32(a);
}
