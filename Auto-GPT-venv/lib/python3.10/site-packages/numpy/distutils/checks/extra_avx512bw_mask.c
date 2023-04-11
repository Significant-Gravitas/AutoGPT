#include <immintrin.h>
/**
 * Test BW mask operations due to:
 *  - MSVC has supported it since vs2019 see,
 *    https://developercommunity.visualstudio.com/content/problem/518298/missing-avx512bw-mask-intrinsics.html
 *  - Clang >= v8.0
 *  - GCC >= v7.1
 */
int main(void)
{
    __mmask64 m64 = _mm512_cmpeq_epi8_mask(_mm512_set1_epi8((char)1), _mm512_set1_epi8((char)1));
    m64 = _kor_mask64(m64, m64);
    m64 = _kxor_mask64(m64, m64);
    m64 = _cvtu64_mask64(_cvtmask64_u64(m64));
    m64 = _mm512_kunpackd(m64, m64);
    m64 = (__mmask64)_mm512_kunpackw((__mmask32)m64, (__mmask32)m64);
    return (int)_cvtmask64_u64(m64);
}
