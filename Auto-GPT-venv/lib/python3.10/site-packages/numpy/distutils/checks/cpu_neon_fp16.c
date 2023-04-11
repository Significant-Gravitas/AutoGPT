#ifdef _MSC_VER
    #include <Intrin.h>
#endif
#include <arm_neon.h>

int main(int argc, char **argv)
{
    short *src = (short*)argv[argc-1];
    float32x4_t v_z4 = vcvt_f32_f16((float16x4_t)vld1_s16(src));
    return (int)vgetq_lane_f32(v_z4, 0);
}
