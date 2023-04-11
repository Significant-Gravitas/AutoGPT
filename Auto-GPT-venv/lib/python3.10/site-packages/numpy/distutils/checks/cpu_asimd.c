#ifdef _MSC_VER
    #include <Intrin.h>
#endif
#include <arm_neon.h>

int main(int argc, char **argv)
{
    float *src = (float*)argv[argc-1];
    float32x4_t v1 = vdupq_n_f32(src[0]), v2 = vdupq_n_f32(src[1]);
    /* MAXMIN */
    int ret  = (int)vgetq_lane_f32(vmaxnmq_f32(v1, v2), 0);
        ret += (int)vgetq_lane_f32(vminnmq_f32(v1, v2), 0);
    /* ROUNDING */
    ret += (int)vgetq_lane_f32(vrndq_f32(v1), 0);
#ifdef __aarch64__
    {
        double *src2 = (double*)argv[argc-1];
        float64x2_t vd1 = vdupq_n_f64(src2[0]), vd2 = vdupq_n_f64(src2[1]);
        /* MAXMIN */
        ret += (int)vgetq_lane_f64(vmaxnmq_f64(vd1, vd2), 0);
        ret += (int)vgetq_lane_f64(vminnmq_f64(vd1, vd2), 0);
        /* ROUNDING */
        ret += (int)vgetq_lane_f64(vrndq_f64(vd1), 0);
    }
#endif
    return ret;
}
