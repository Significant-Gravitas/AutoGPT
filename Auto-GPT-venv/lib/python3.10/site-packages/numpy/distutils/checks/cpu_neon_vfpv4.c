#ifdef _MSC_VER
    #include <Intrin.h>
#endif
#include <arm_neon.h>

int main(int argc, char **argv)
{
    float *src = (float*)argv[argc-1];
    float32x4_t v1 = vdupq_n_f32(src[0]);
    float32x4_t v2 = vdupq_n_f32(src[1]);
    float32x4_t v3 = vdupq_n_f32(src[2]);
    int ret = (int)vgetq_lane_f32(vfmaq_f32(v1, v2, v3), 0);
#ifdef __aarch64__
    double *src2 = (double*)argv[argc-2];
    float64x2_t vd1 = vdupq_n_f64(src2[0]);
    float64x2_t vd2 = vdupq_n_f64(src2[1]);
    float64x2_t vd3 = vdupq_n_f64(src2[2]);
    ret += (int)vgetq_lane_f64(vfmaq_f64(vd1, vd2, vd3), 0);
#endif
    return ret;
}
