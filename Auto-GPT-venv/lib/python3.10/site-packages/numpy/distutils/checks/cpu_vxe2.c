#if (__VEC__ < 10303) || (__ARCH__ < 13)
    #error VXE2 not supported
#endif

#include <vecintrin.h>

int main(int argc, char **argv)
{
    int val;
    __vector signed short large = { 'a', 'b', 'c', 'a', 'g', 'h', 'g', 'o' };
    __vector signed short search = { 'g', 'h', 'g', 'o' };
    __vector unsigned char len = { 0 };
    __vector unsigned char res = vec_search_string_cc(large, search, len, &val);
    __vector float x = vec_xl(argc, (float*)argv);
    __vector int i = vec_signed(x);

    i = vec_srdb(vec_sldb(i, i, 2), i, 3);
    val += (int)vec_extract(res, 1);
    val += vec_extract(i, 0);
    return val;
}
