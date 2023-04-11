/**
 * Testing ASM VSX register number fixer '%x<n>'
 *
 * old versions of CLANG doesn't support %x<n> in the inline asm template
 * which fixes register number when using any of the register constraints wa, wd, wf.
 *
 * xref:
 * - https://bugs.llvm.org/show_bug.cgi?id=31837
 * - https://gcc.gnu.org/onlinedocs/gcc/Machine-Constraints.html
 */
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
    float z4[] = {0, 0, 0, 0};
    signed int zout[] = {0, 0, 0, 0};

    __vector float vz4 = vsx_ld(0, z4);
    __vector signed int asm_ret = vsx_ld(0, zout);

    __asm__ ("xvcvspsxws %x0,%x1" : "=wa" (vz4) : "wa" (asm_ret));

    vsx_st(asm_ret, 0, zout);
    return zout[0];
}
