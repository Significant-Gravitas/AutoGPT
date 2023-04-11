#ifdef _MSC_VER
    #include <Intrin.h>
#endif
#include <arm_neon.h>

int main(int argc, char **argv)
{
    // passing from untraced pointers to avoid optimizing out any constants
    // so we can test against the linker.
    float *src = (float*)argv[argc-1];
    float32x4_t v1 = vdupq_n_f32(src[0]), v2 = vdupq_n_f32(src[1]);
    int ret = (int)vgetq_lane_f32(vmulq_f32(v1, v2), 0);
#ifdef __aarch64__
    double *src2 = (double*)argv[argc-2];
    float64x2_t vd1 = vdupq_n_f64(src2[0]), vd2 = vdupq_n_f64(src2[1]);
    ret += (int)vgetq_lane_f64(vmulq_f64(vd1, vd2), 0);
#endif
    return ret;
}
