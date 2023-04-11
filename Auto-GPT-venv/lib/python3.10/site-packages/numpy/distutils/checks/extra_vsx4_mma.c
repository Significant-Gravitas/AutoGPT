#ifndef __VSX__
    #error "VSX is not supported"
#endif
#include <altivec.h>

typedef __vector float fv4sf_t;
typedef __vector unsigned char vec_t;

int main(void)
{
    __vector_quad acc0;
    float a[4] = {0,1,2,3};
    float b[4] = {0,1,2,3};
    vec_t *va = (vec_t *) a;
    vec_t *vb = (vec_t *) b;
    __builtin_mma_xvf32ger(&acc0, va[0], vb[0]);
    fv4sf_t result[4];
    __builtin_mma_disassemble_acc((void *)result, &acc0);
    fv4sf_t c0 = result[0];
    return (int)((float*)&c0)[0];
}
