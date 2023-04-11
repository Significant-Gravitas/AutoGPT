#include <immintrin.h>
#ifdef _MSC_VER
    #include <ammintrin.h>
#else
    #include <x86intrin.h>
#endif

int main(void)
{
    __m128i a = _mm_comge_epu32(_mm_setzero_si128(), _mm_setzero_si128());
    return _mm_cvtsi128_si32(a);
}
