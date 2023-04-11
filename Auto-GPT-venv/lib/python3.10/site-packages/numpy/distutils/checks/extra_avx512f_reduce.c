#include <immintrin.h>
/**
 * The following intrinsics don't have direct native support but compilers
 * tend to emulate them.
 * They're usually supported by gcc >= 7.1, clang >= 4 and icc >= 19
 */
int main(void)
{
    __m512  one_ps = _mm512_set1_ps(1.0f);
    __m512d one_pd = _mm512_set1_pd(1.0);
    __m512i one_i64 = _mm512_set1_epi64(1);
    // add
    float sum_ps  = _mm512_reduce_add_ps(one_ps);
    double sum_pd = _mm512_reduce_add_pd(one_pd);
    int sum_int   = (int)_mm512_reduce_add_epi64(one_i64);
        sum_int  += (int)_mm512_reduce_add_epi32(one_i64);
    // mul
    sum_ps  += _mm512_reduce_mul_ps(one_ps);
    sum_pd  += _mm512_reduce_mul_pd(one_pd);
    sum_int += (int)_mm512_reduce_mul_epi64(one_i64);
    sum_int += (int)_mm512_reduce_mul_epi32(one_i64);
    // min
    sum_ps  += _mm512_reduce_min_ps(one_ps);
    sum_pd  += _mm512_reduce_min_pd(one_pd);
    sum_int += (int)_mm512_reduce_min_epi32(one_i64);
    sum_int += (int)_mm512_reduce_min_epu32(one_i64);
    sum_int += (int)_mm512_reduce_min_epi64(one_i64);
    // max
    sum_ps  += _mm512_reduce_max_ps(one_ps);
    sum_pd  += _mm512_reduce_max_pd(one_pd);
    sum_int += (int)_mm512_reduce_max_epi32(one_i64);
    sum_int += (int)_mm512_reduce_max_epu32(one_i64);
    sum_int += (int)_mm512_reduce_max_epi64(one_i64);
    // and
    sum_int += (int)_mm512_reduce_and_epi32(one_i64);
    sum_int += (int)_mm512_reduce_and_epi64(one_i64);
    // or
    sum_int += (int)_mm512_reduce_or_epi32(one_i64);
    sum_int += (int)_mm512_reduce_or_epi64(one_i64);
    return (int)sum_ps + (int)sum_pd + sum_int;
}
