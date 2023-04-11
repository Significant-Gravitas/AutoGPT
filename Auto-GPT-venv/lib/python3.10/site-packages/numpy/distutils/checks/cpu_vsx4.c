#ifndef __VSX__
    #error "VSX is not supported"
#endif
#include <altivec.h>

typedef __vector unsigned int v_uint32x4;

int main(void)
{
    v_uint32x4 v1 = (v_uint32x4){2, 4, 8, 16};
    v_uint32x4 v2 = (v_uint32x4){2, 2, 2, 2};
    v_uint32x4 v3 = vec_mod(v1, v2);
    return (int)vec_extractm(v3);
}
