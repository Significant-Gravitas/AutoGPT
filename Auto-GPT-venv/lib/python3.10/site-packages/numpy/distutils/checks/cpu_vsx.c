#ifndef __VSX__
    #error "VSX is not supported"
#endif
#include <altivec.h>

#if (defined(__GNUC__) && !defined(vec_xl)) || (defined(__clang__) && !defined(__IBMC__))
    #define vsx_ld  vec_vsx_ld
    #define vsx_st  vec_vsx_st
#else
    #define vsx_ld  vec_xl
    #define vsx_st  vec_xst
#endif

int main(void)
{
    unsigned int zout[4];
    unsigned int z4[] = {0, 0, 0, 0};
    __vector unsigned int v_z4 = vsx_ld(0, z4);
    vsx_st(v_z4, 0, zout);
    return zout[0];
}
