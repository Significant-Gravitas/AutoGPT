#if defined(DETECT_FEATURES) && defined(__INTEL_COMPILER)
    /*
     * Unlike GCC and CLANG, Intel Compiler exposes all supported intrinsics,
     * whether or not the build options for those features are specified.
     * Therefore, we must test #definitions of CPU features when option native/host
     * is enabled via `--cpu-baseline` or through env var `CFLAGS` otherwise
     * the test will be broken and leads to enable all possible features.
     */
    #if !defined(__FMA__) && !defined(__AVX2__)
        #error "HOST/ARCH doesn't support FMA3"
    #endif
#endif

#include <xmmintrin.h>
#include <immintrin.h>

int main(int argc, char **argv)
{
    __m256 a = _mm256_loadu_ps((const float*)argv[argc-1]);
           a = _mm256_fmadd_ps(a, a, a);
    return (int)_mm_cvtss_f32(_mm256_castps256_ps128(a));
}
