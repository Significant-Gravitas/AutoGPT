#if (__VEC__ < 10302) || (__ARCH__ < 12)
    #error VXE not supported
#endif

#include <vecintrin.h>
int main(int argc, char **argv)
{
    __vector float x = vec_nabs(vec_xl(argc, (float*)argv));
    __vector float y = vec_load_len((float*)argv, (unsigned int)argc);
    
    x = vec_round(vec_ceil(x) + vec_floor(y));
    __vector bool int m = vec_cmpge(x, y);
    x = vec_sel(x, y, m);

    // need to test the existence of intrin "vflls" since vec_doublee
    // is vec_doublee maps to wrong intrin "vfll".
    // see https://gcc.gnu.org/bugzilla/show_bug.cgi?id=100871
#if defined(__GNUC__) && !defined(__clang__)
    __vector long long i = vec_signed(__builtin_s390_vflls(x));
#else
    __vector long long i = vec_signed(vec_doublee(x));
#endif

    return (int)vec_extract(i, 0);
}
