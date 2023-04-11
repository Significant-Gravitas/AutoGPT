#ifdef _MSC_VER
    #include <Intrin.h>
#endif
#include <arm_neon.h>

int main(int argc, char **argv)
{
    unsigned char *src = (unsigned char*)argv[argc-1];
    uint8x16_t v1 = vdupq_n_u8(src[0]), v2 = vdupq_n_u8(src[1]);
    uint32x4_t va = vdupq_n_u32(3);
    int ret = (int)vgetq_lane_u32(vdotq_u32(va, v1, v2), 0);
#ifdef __aarch64__
    ret += (int)vgetq_lane_u32(vdotq_laneq_u32(va, v1, v2, 0), 0);
#endif
    return ret;
}
