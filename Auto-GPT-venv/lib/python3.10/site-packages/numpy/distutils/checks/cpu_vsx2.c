#ifndef __VSX__
    #error "VSX is not supported"
#endif
#include <altivec.h>

typedef __vector unsigned long long v_uint64x2;

int main(void)
{
    v_uint64x2 z2 = (v_uint64x2){0, 0};
    z2 = (v_uint64x2)vec_cmpeq(z2, z2);
    return (int)vec_extract(z2, 0);
}
