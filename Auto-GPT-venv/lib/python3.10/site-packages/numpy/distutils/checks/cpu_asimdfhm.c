#ifdef _MSC_VER
    #include <Intrin.h>
#endif
#include <arm_neon.h>

int main(int argc, char **argv)
{
    float16_t *src = (float16_t*)argv[argc-1];
    float *src2 = (float*)argv[argc-2];
    float16x8_t vhp  = vdupq_n_f16(src[0]);
    float16x4_t vlhp = vdup_n_f16(src[1]);
    float32x4_t vf   = vdupq_n_f32(src2[0]);
    float32x2_t vlf  = vdup_n_f32(src2[1]);

    int ret  = (int)vget_lane_f32(vfmlal_low_f16(vlf, vlhp, vlhp), 0);
        ret += (int)vgetq_lane_f32(vfmlslq_high_f16(vf, vhp, vhp), 0);

    return ret;
}
