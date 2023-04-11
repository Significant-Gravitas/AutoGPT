#if (__VEC__ < 10301) || (__ARCH__ < 11)
    #error VX not supported
#endif

#include <vecintrin.h>
int main(int argc, char **argv)
{
    __vector double x = vec_abs(vec_xl(argc, (double*)argv));
    __vector double y = vec_load_len((double*)argv, (unsigned int)argc);

    x = vec_round(vec_ceil(x) + vec_floor(y));
    __vector bool long long m = vec_cmpge(x, y);
    __vector long long i = vec_signed(vec_sel(x, y, m));

    return (int)vec_extract(i, 0);
}
