#include <immintrin.h>
/**
 * Test DQ mask operations due to:
 *  - MSVC has supported it since vs2019 see,
 *    https://developercommunity.visualstudio.com/content/problem/518298/missing-avx512bw-mask-intrinsics.html
 *  - Clang >= v8.0
 *  - GCC >= v7.1
 */
int main(void)
{
    __mmask8 m8 = _mm512_cmpeq_epi64_mask(_mm512_set1_epi64(1), _mm512_set1_epi64(1));
    m8 = _kor_mask8(m8, m8);
    m8 = _kxor_mask8(m8, m8);
    m8 = _cvtu32_mask8(_cvtmask8_u32(m8));
    return (int)_cvtmask8_u32(m8);
}
