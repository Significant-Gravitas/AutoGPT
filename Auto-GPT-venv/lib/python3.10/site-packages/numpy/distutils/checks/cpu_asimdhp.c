#ifdef _MSC_VER
    #include <Intrin.h>
#endif
#include <arm_neon.h>

int main(int argc, char **argv)
{
    float16_t *src = (float16_t*)argv[argc-1];
    float16x8_t vhp  = vdupq_n_f16(src[0]);
    float16x4_t vlhp = vdup_n_f16(src[1]);

    int ret  =  (int)vgetq_lane_f16(vabdq_f16(vhp, vhp), 0);
        ret  += (int)vget_lane_f16(vabd_f16(vlhp, vlhp), 0);
    return ret;
}
