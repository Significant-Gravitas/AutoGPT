#ifndef __VSX__
    #error "VSX is not supported"
#endif
#include <altivec.h>

typedef __vector unsigned int v_uint32x4;

int main(void)
{
    v_uint32x4 z4 = (v_uint32x4){0, 0, 0, 0};
    z4 = vec_absd(z4, z4);
    return (int)vec_extract(z4, 0);
}
