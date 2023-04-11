#if defined(DETECT_FEATURES) && defined(__INTEL_COMPILER)
    /*
     * Unlike GCC and CLANG, Intel Compiler exposes all supported intrinsics,
     * whether or not the build options for those features are specified.
     * Therefore, we must test #definitions of CPU features when option native/host
     * is enabled via `--cpu-baseline` or through env var `CFLAGS` otherwise
     * the test will be broken and leads to enable all possible features.
     */
    #ifndef __F16C__
        #error "HOST/ARCH doesn't support F16C"
    #endif
#endif

#include <emmintrin.h>
#include <immintrin.h>

int main(int argc, char **argv)
{
    __m128 a  = _mm_cvtph_ps(_mm_loadu_si128((const __m128i*)argv[argc-1]));
    __m256 a8 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)argv[argc-2]));
    return (int)(_mm_cvtss_f32(a) + _mm_cvtss_f32(_mm256_castps256_ps128(a8)));
}
