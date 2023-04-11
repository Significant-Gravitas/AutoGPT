#include <immintrin.h>
#ifdef _MSC_VER
    #include <ammintrin.h>
#else
    #include <x86intrin.h>
#endif

int main(int argc, char **argv)
{
    __m256 a = _mm256_loadu_ps((const float*)argv[argc-1]);
           a = _mm256_macc_ps(a, a, a);
    return (int)_mm_cvtss_f32(_mm256_castps256_ps128(a));
}
