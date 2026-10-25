// NEON kernels for AArch64. A lane-by-lane port of the SSE2 kernels in sse2.cpp: the same loop
// bounds, the same edge handling, the same 16 bit arithmetic in the warp, so the results are
// bit-identical to the SSE2 and the C code (the tests compare opt=True against opt=False).

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <arm_neon.h>

#include "warp_edge.h"


#ifdef _MSC_VER
#define FORCE_INLINE __forceinline
#else
#define FORCE_INLINE inline __attribute__((always_inline))
#endif


// Vector operations per pixel type, 16 bytes per vector (16 x uint8_t or 8 x uint16_t).
template <typename PixelType>
struct Vec;

template <>
struct Vec<uint8_t> {
    typedef uint8x16_t type;

    static FORCE_INLINE type load(const uint8_t *p) { return vld1q_u8(p); }
    static FORCE_INLINE void store(uint8_t *p, type v) { vst1q_u8(p, v); }
    static FORCE_INLINE type set1(int a) { return vdupq_n_u8((uint8_t)a); }
    static FORCE_INLINE type avg(type a, type b) { return vrhaddq_u8(a, b); } // (a + b + 1) >> 1
    static FORCE_INLINE type absdiff(type a, type b) { return vabdq_u8(a, b); }
    static FORCE_INLINE type adds(type a, type b) { return vqaddq_u8(a, b); }
    static FORCE_INLINE type max(type a, type b) { return vmaxq_u8(a, b); }
    static FORCE_INLINE type min(type a, type b) { return vminq_u8(a, b); }

    // Right edge of the horizontal r6 blur: the vector covers the last 16 pixels of the row,
    // but only the last six were computed with the right-edge formula. Pixels 8 and 9 keep
    // the value from the middle pass, pixels 0..7 are not stored at all (same as the SSE2 code).
    static FORCE_INLINE void store_right_edge(uint8_t *dstp, type v) {
        uint16_t keep;
        memcpy(&keep, dstp + 8, sizeof(keep));
        uint16x4_t high = vreinterpret_u16_u8(vget_high_u8(v));
        high = vset_lane_u16(keep, high, 0);
        vst1_u8(dstp + 8, vreinterpret_u8_u16(high));
    }
};

template <>
struct Vec<uint16_t> {
    typedef uint16x8_t type;

    static FORCE_INLINE type load(const uint16_t *p) { return vld1q_u16(p); }
    static FORCE_INLINE void store(uint16_t *p, type v) { vst1q_u16(p, v); }
    static FORCE_INLINE type set1(int a) { return vdupq_n_u16((uint16_t)a); }
    static FORCE_INLINE type avg(type a, type b) { return vrhaddq_u16(a, b); }
    static FORCE_INLINE type absdiff(type a, type b) { return vabdq_u16(a, b); }
    static FORCE_INLINE type adds(type a, type b) { return vqaddq_u16(a, b); }
    static FORCE_INLINE type max(type a, type b) { return vmaxq_u16(a, b); }
    static FORCE_INLINE type min(type a, type b) { return vminq_u16(a, b); }

    // The vector covers the last 8 pixels of the row; the first two keep the middle pass's value.
    static FORCE_INLINE void store_right_edge(uint16_t *dstp, type v) {
        v = vsetq_lane_u16(dstp[0], v, 0);
        v = vsetq_lane_u16(dstp[1], v, 1);
        vst1q_u16(dstp, v);
    }
};


template <typename PixelType>
static FORCE_INLINE void sobel_vector_neon(const PixelType *srcp, PixelType *dstp, int stride, const typename Vec<PixelType>::type &th, const typename Vec<PixelType>::type &pixel_max) {
    typedef Vec<PixelType> V;
    typedef typename V::type vec;

    vec a11, a21, a31,
        a12,      a32,
        a13, a23, a33;

    a11 = V::load(srcp - stride - 1);
    a21 = V::load(srcp - stride);
    a31 = V::load(srcp - stride + 1);

    a12 = V::load(srcp - 1);
    a32 = V::load(srcp + 1);

    a13 = V::load(srcp + stride - 1);
    a23 = V::load(srcp + stride);
    a33 = V::load(srcp + stride + 1);

    vec avg_up    = V::avg(a21, V::avg(a11, a31));
    vec avg_down  = V::avg(a23, V::avg(a13, a33));
    vec avg_left  = V::avg(a12, V::avg(a13, a11));
    vec avg_right = V::avg(a32, V::avg(a33, a31));

    vec abs_v = V::absdiff(avg_up, avg_down);
    vec abs_h = V::absdiff(avg_left, avg_right);

    // 8 bit: the saturating adds cannot exceed pixel_max, so the min is only needed for 9..16 bit.
    vec absolute = V::adds(abs_v, abs_h);
    if (sizeof(PixelType) == 2)
        absolute = V::min(absolute, pixel_max);

    vec abs_max = V::max(abs_h, abs_v);

    absolute = V::adds(absolute, abs_max);
    if (sizeof(PixelType) == 2)
        absolute = V::min(absolute, pixel_max);

    vec absolute2 = V::adds(absolute, absolute);
    if (sizeof(PixelType) == 2)
        absolute2 = V::min(absolute2, pixel_max);

    absolute = V::adds(absolute2, absolute);
    if (sizeof(PixelType) == 2)
        absolute = V::min(absolute, pixel_max);

    absolute = V::adds(absolute, absolute);
    if (sizeof(PixelType) == 2)
        absolute = V::min(absolute, pixel_max);

    V::store(dstp, V::min(absolute, th));
}


template <typename PixelType>
static void sobel_neon(const uint8_t *srcp8, uint8_t *dstp8, int stride, int width, int height, int thresh, int bits_per_sample) {
    typedef Vec<PixelType> V;

    const PixelType *srcp = (const PixelType *)srcp8;
    PixelType *dstp = (PixelType *)dstp8;

    stride /= sizeof(PixelType);

    typename V::type pixel_max = V::set1((1 << bits_per_sample) - 1);

    const int pixels_in_vec = 16 / sizeof(PixelType);

    PixelType *dstp_orig = dstp;

    srcp += stride;
    dstp += stride;

    typename V::type th = V::set1(thresh);

    int width_simd = (width & ~(pixels_in_vec - 1)) + 2;
    if (width_simd > stride)
        width_simd -= pixels_in_vec;

    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width_simd - 1; x += pixels_in_vec)
            sobel_vector_neon<PixelType>(srcp + x, dstp + x, stride, th, pixel_max);

        // Last vector, clamped so a narrow row (width < pixels_in_vec + 2) does not start before the row.
        if (width + 2 > width_simd) {
            int x = std::max(width - pixels_in_vec - 1, 1);
            sobel_vector_neon<PixelType>(srcp + x, dstp + x, stride, th, pixel_max);
        }

        dstp[0] = dstp[1];
        dstp[width - 1] = dstp[width - 2];

        srcp += stride;
        dstp += stride;
    }

    memcpy(dstp_orig, dstp_orig + stride, width * sizeof(PixelType));
    memcpy(dstp, dstp - stride, width * sizeof(PixelType));
}


template <typename PixelType>
static FORCE_INLINE void blur_r6_h_left_neon(const PixelType *srcp, PixelType *dstp) {
    typedef Vec<PixelType> V;
    typedef typename V::type vec;

    vec avg12 = V::avg(V::load(srcp + 1), V::load(srcp + 2));
    vec avg34 = V::avg(V::load(srcp + 3), V::load(srcp + 4));
    vec avg56 = V::avg(V::load(srcp + 5), V::load(srcp + 6));

    vec avg012 = V::avg(V::load(srcp), avg12);
    vec avg3456 = V::avg(avg34, avg56);
    vec avg0123456 = V::avg(avg012, avg3456);
    vec avg = V::avg(avg012, avg0123456);

    V::store(dstp, avg);
}


template <typename PixelType>
static FORCE_INLINE void blur_r6_h_middle_neon(const PixelType *srcp, PixelType *dstp) {
    typedef Vec<PixelType> V;
    typedef typename V::type vec;

    vec avg11 = V::avg(V::load(srcp - 1), V::load(srcp + 1));
    vec avg22 = V::avg(V::load(srcp - 2), V::load(srcp + 2));
    vec avg33 = V::avg(V::load(srcp - 3), V::load(srcp + 3));
    vec avg44 = V::avg(V::load(srcp - 4), V::load(srcp + 4));
    vec avg55 = V::avg(V::load(srcp - 5), V::load(srcp + 5));
    vec avg66 = V::avg(V::load(srcp - 6), V::load(srcp + 6));

    vec avg12 = V::avg(avg11, avg22);
    vec avg34 = V::avg(avg33, avg44);
    vec avg56 = V::avg(avg55, avg66);
    vec avg012 = V::avg(V::load(srcp), avg12);
    vec avg3456 = V::avg(avg34, avg56);
    vec avg0123456 = V::avg(avg012, avg3456);
    vec avg = V::avg(avg012, avg0123456);

    V::store(dstp, avg);
}


template <typename PixelType>
static FORCE_INLINE void blur_r6_h_right_neon(const PixelType *srcp, PixelType *dstp) {
    typedef Vec<PixelType> V;
    typedef typename V::type vec;

    vec avg12 = V::avg(V::load(srcp - 1), V::load(srcp - 2));
    vec avg34 = V::avg(V::load(srcp - 3), V::load(srcp - 4));
    vec avg56 = V::avg(V::load(srcp - 5), V::load(srcp - 6));

    vec avg012 = V::avg(V::load(srcp), avg12);
    vec avg3456 = V::avg(avg34, avg56);
    vec avg0123456 = V::avg(avg012, avg3456);
    vec avg = V::avg(avg012, avg0123456);

    // This is the right edge. Only the highest six pixels are needed.
    V::store_right_edge(dstp, avg);
}


template <typename PixelType>
static FORCE_INLINE void blur_r6_v_top_neon(const PixelType *srcp, PixelType *dstp, int stride) {
    typedef Vec<PixelType> V;
    typedef typename V::type vec;

    vec l0 = V::load(srcp);
    vec l1 = V::load(srcp + stride);
    vec l2 = V::load(srcp + stride * 2);
    vec l3 = V::load(srcp + stride * 3);
    vec l4 = V::load(srcp + stride * 4);
    vec l5 = V::load(srcp + stride * 5);
    vec l6 = V::load(srcp + stride * 6);

    vec avg12 = V::avg(l1, l2);
    vec avg34 = V::avg(l3, l4);
    vec avg56 = V::avg(l5, l6);

    vec avg3456 = V::avg(avg34, avg56);
    vec avg012 = V::avg(l0, avg12);
    vec avg0123456 = V::avg(avg012, avg3456);
    vec avg = V::avg(avg012, avg0123456);

    V::store(dstp, avg);
}


template <typename PixelType>
static FORCE_INLINE void blur_r6_v_middle_neon(const PixelType *srcp, PixelType *dstp, int stride) {
    typedef Vec<PixelType> V;
    typedef typename V::type vec;

    vec m6 = V::load(srcp - stride * 6);
    vec m5 = V::load(srcp - stride * 5);
    vec m4 = V::load(srcp - stride * 4);
    vec m3 = V::load(srcp - stride * 3);
    vec m2 = V::load(srcp - stride * 2);
    vec m1 = V::load(srcp - stride);
    vec l0 = V::load(srcp);
    vec l1 = V::load(srcp + stride);
    vec l2 = V::load(srcp + stride * 2);
    vec l3 = V::load(srcp + stride * 3);
    vec l4 = V::load(srcp + stride * 4);
    vec l5 = V::load(srcp + stride * 5);
    vec l6 = V::load(srcp + stride * 6);

    vec avg11 = V::avg(m1, l1);
    vec avg22 = V::avg(m2, l2);
    vec avg33 = V::avg(m3, l3);
    vec avg44 = V::avg(m4, l4);
    vec avg55 = V::avg(m5, l5);
    vec avg66 = V::avg(m6, l6);

    vec avg12 = V::avg(avg11, avg22);
    vec avg34 = V::avg(avg33, avg44);
    vec avg56 = V::avg(avg55, avg66);
    vec avg012 = V::avg(l0, avg12);
    vec avg3456 = V::avg(avg34, avg56);
    vec avg0123456 = V::avg(avg012, avg3456);
    vec avg = V::avg(avg012, avg0123456);

    V::store(dstp, avg);
}


template <typename PixelType>
static FORCE_INLINE void blur_r6_v_bottom_neon(const PixelType *srcp, PixelType *dstp, int stride) {
    typedef Vec<PixelType> V;
    typedef typename V::type vec;

    vec m6 = V::load(srcp - stride * 6);
    vec m5 = V::load(srcp - stride * 5);
    vec m4 = V::load(srcp - stride * 4);
    vec m3 = V::load(srcp - stride * 3);
    vec m2 = V::load(srcp - stride * 2);
    vec m1 = V::load(srcp - stride);
    vec l0 = V::load(srcp);

    vec avg12 = V::avg(m1, m2);
    vec avg34 = V::avg(m3, m4);
    vec avg56 = V::avg(m5, m6);
    vec avg012 = V::avg(l0, avg12);
    vec avg3456 = V::avg(avg34, avg56);
    vec avg0123456 = V::avg(avg012, avg3456);
    vec avg = V::avg(avg012, avg0123456);

    V::store(dstp, avg);
}


template <typename PixelType>
static void blur_r6_neon(uint8_t *mask8, uint8_t *temp8, int stride, int width, int height) {
    // Horizontal blur from mask to temp.
    // Vertical blur from temp back to mask.

    PixelType *mask = (PixelType *)mask8;
    PixelType *temp = (PixelType *)temp8;

    stride /= sizeof(PixelType);

    const int pixels_in_vec = 16 / sizeof(PixelType);

    int width_simd = (width & ~(pixels_in_vec - 1)) + 12;
    if (width_simd > stride)
        width_simd -= pixels_in_vec;

    PixelType *mask_orig = mask;
    PixelType *temp_orig = temp;

    // Horizontal blur.

    for (int y = 0; y < height; y++) {
        blur_r6_h_left_neon<PixelType>(mask, temp);

        for (int x = 6; x < width_simd - 6; x += pixels_in_vec)
            blur_r6_h_middle_neon<PixelType>(mask + x, temp + x);

        // Last middle vector, clamped to the first middle pixel: starting below 6 would overwrite
        // the left edge computed above with the middle formula.
        if (width + 12 > width_simd) {
            int x = std::max(width - pixels_in_vec - 6, 6);
            blur_r6_h_middle_neon<PixelType>(mask + x, temp + x);
        }

        blur_r6_h_right_neon<PixelType>(mask + width - pixels_in_vec, temp + width - pixels_in_vec);

        mask += stride;
        temp += stride;
    }


    // Vertical blur.

    width_simd = width & ~(pixels_in_vec - 1);

    mask = mask_orig;
    temp = temp_orig;
    int y;

    for (y = 0; y < 6; y++) {
        for (int x = 0; x < width_simd; x += pixels_in_vec)
            blur_r6_v_top_neon<PixelType>(temp + x, mask + x, stride);

        if (width > width_simd)
            blur_r6_v_top_neon<PixelType>(temp + width - pixels_in_vec, mask + width - pixels_in_vec, stride);

        mask += stride;
        temp += stride;
    }

    for ( ; y < height - 6; y++) {
        for (int x = 0; x < width_simd; x += pixels_in_vec)
            blur_r6_v_middle_neon<PixelType>(temp + x, mask + x, stride);

        if (width > width_simd)
            blur_r6_v_middle_neon<PixelType>(temp + width - pixels_in_vec, mask + width - pixels_in_vec, stride);

        mask += stride;
        temp += stride;
    }

    for ( ; y < height; y++) {
        for (int x = 0; x < width_simd; x += pixels_in_vec)
            blur_r6_v_bottom_neon<PixelType>(temp + x, mask + x, stride);

        if (width > width_simd)
            blur_r6_v_bottom_neon<PixelType>(temp + width - pixels_in_vec, mask + width - pixels_in_vec, stride);

        mask += stride;
        temp += stride;
    }
}


template <typename PixelType>
static FORCE_INLINE void blur_r2_h_neon(const PixelType *srcp, PixelType *dstp) {
    typedef Vec<PixelType> V;
    typedef typename V::type vec;

    vec avg1 = V::avg(V::load(srcp - 1), V::load(srcp + 1));
    vec avg2 = V::avg(V::load(srcp - 2), V::load(srcp + 2));
    vec middle = V::load(srcp);
    vec avg = V::avg(avg2, middle);
    avg = V::avg(avg, middle);
    avg = V::avg(avg, avg1);

    V::store(dstp, avg);
}


template <typename PixelType>
static FORCE_INLINE void blur_r2_v_neon(const PixelType *srcp, PixelType *dstp, int stride_p2, int stride_p1, int stride_n1, int stride_n2) {
    typedef Vec<PixelType> V;
    typedef typename V::type vec;

    vec m2 = V::load(srcp + stride_p2);
    vec m1 = V::load(srcp + stride_p1);
    vec l0 = V::load(srcp);
    vec l1 = V::load(srcp + stride_n1);
    vec l2 = V::load(srcp + stride_n2);

    vec avg1 = V::avg(m1, l1);
    vec avg2 = V::avg(m2, l2);
    vec avg = V::avg(avg2, l0);
    avg = V::avg(avg, l0);
    avg = V::avg(avg, avg1);

    V::store(dstp, avg);
}


template <typename PixelType>
static void blur_r2_neon(uint8_t *mask8, uint8_t *temp8, int stride, int width, int height) {
    // Horizontal blur from mask to temp.
    // Vertical blur from temp back to mask.

    PixelType *mask = (PixelType *)mask8;
    PixelType *temp = (PixelType *)temp8;

    stride /= sizeof(PixelType);

    const int pixels_in_vec = 16 / sizeof(PixelType);

    int width_simd = (width & ~(pixels_in_vec - 1)) + 4;
    if (width_simd > stride)
        width_simd -= pixels_in_vec;

    PixelType *mask_orig = mask;
    PixelType *temp_orig = temp;

    // Horizontal blur.

    for (int y = 0; y < height; y++) {
        int avg, avg1, avg2;

        avg1 = (mask[0] + mask[1] + 1) >> 1;
        avg2 = (mask[0] + mask[2] + 1) >> 1;
        avg = (avg2 + mask[0] + 1) >> 1;
        avg = (avg + mask[0] + 1) >> 1;
        avg = (avg + avg1 + 1) >> 1;

        temp[0] = avg;

        avg1 = (mask[0] + mask[2] + 1) >> 1;
        avg2 = (mask[0] + mask[3] + 1) >> 1;
        avg = (avg2 + mask[1] + 1) >> 1;
        avg = (avg + mask[1] + 1) >> 1;
        avg = (avg + avg1 + 1) >> 1;

        temp[1] = avg;

        for (int x = 2; x < width_simd - 2; x += pixels_in_vec)
            blur_r2_h_neon<PixelType>(mask + x, temp + x);

        // Last middle vector, clamped to the first middle pixel (see blur_r6).
        if (width + 4 > width_simd) {
            int x = std::max(width - pixels_in_vec - 2, 2);
            blur_r2_h_neon<PixelType>(mask + x, temp + x);
        }

        avg1 = (mask[width - 3] + mask[width - 1] + 1) >> 1;
        avg2 = (mask[width - 4] + mask[width - 1] + 1) >> 1;
        avg = (avg2 + mask[width - 2] + 1) >> 1;
        avg = (avg + mask[width - 2] + 1) >> 1;
        avg = (avg + avg1 + 1) >> 1;

        temp[width - 2] = avg;

        avg1 = (mask[width - 2] + mask[width - 1] + 1) >> 1;
        avg2 = (mask[width - 3] + mask[width - 1] + 1) >> 1;
        avg = (avg2 + mask[width - 1] + 1) >> 1;
        avg = (avg + mask[width - 1] + 1) >> 1;
        avg = (avg + avg1 + 1) >> 1;

        temp[width - 1] = avg;

        mask += stride;
        temp += stride;
    }


    // Vertical blur.

    width_simd = width & ~(pixels_in_vec - 1);

    mask = mask_orig;
    temp = temp_orig;

    for (int y = 0; y < height; y++) {
        int stride_p1 = y ? -stride : 0;
        int stride_p2 = y > 1 ? stride_p1 * 2 : stride_p1;
        int stride_n1 = y < height - 1 ? stride : 0;
        int stride_n2 = y < height - 2 ? stride_n1 * 2 : stride_n1;

        for (int x = 0; x < width_simd; x += pixels_in_vec)
            blur_r2_v_neon<PixelType>(temp + x, mask + x, stride_p2, stride_p1, stride_n1, stride_n2);

        if (width > width_simd)
            blur_r2_v_neon<PixelType>(temp + width - pixels_in_vec, mask + width - pixels_in_vec, stride_p2, stride_p1, stride_n1, stride_n2);

        mask += stride;
        temp += stride;
    }
}


// Saturating conversion to int16, like the _mm_packs_epi32 the SSE2 code uses for its constants.
static FORCE_INLINE int16_t sat16(int v) {
    return (int16_t)std::min(std::max(v, -32768), 32767);
}


// High 16 bits of the signed 16x16 -> 32 bit products (_mm_mulhi_epi16).
static FORCE_INLINE int16x8_t mulhi_s16(int16x8_t a, int16x8_t b) {
    int32x4_t lo = vmull_s16(vget_low_s16(a), vget_low_s16(b));
    int32x4_t hi = vmull_s16(vget_high_s16(a), vget_high_s16(b));
    return vcombine_s16(vshrn_n_s32(lo, 16), vshrn_n_s32(hi, 16));
}


static FORCE_INLINE int16x8_t load_u8_as_s16(const uint8_t *p) {
    return vreinterpretq_s16_u16(vmovl_u8(vld1_u8(p)));
}


// Warps eight 8 bit pixels starting at x. Mirrors warp_mmword_u8_sse2: the displacement is
// computed in 16 bit lanes, the four source pixels per output pixel are gathered with scalar
// loads, the bilinear interpolation runs in 16 bit lanes again.
template <int SMAGL>
static FORCE_INLINE void warp_vector_u8_neon(const uint8_t *srcp, const uint8_t *edgep, uint8_t *dstp, int src_stride, int edge_stride, int height, int x, int y, const int16x8_t &depth, const int16x8_t &x_limit_min, const int16x8_t &x_limit_max, const int16x8_t &y_limit_min, const int16x8_t &y_limit_max) {
    const int SMAG = 1 << SMAGL;

    const int16x8_t word_64 = vdupq_n_s16(64);
    const int16x8_t word_127 = vdupq_n_s16(127);
    const int16x8_t word_128 = vdupq_n_s16(128);

    // calculate displacement

    int16x8_t above = load_u8_as_s16(edgep + x - (y ? edge_stride : 0));
    int16x8_t below = load_u8_as_s16(edgep + x + (y < height - 1 ? edge_stride : 0));

    int16x8_t left = load_u8_as_s16(edgep + x - 1);
    int16x8_t right = load_u8_as_s16(edgep + x + 1);

    int16x8_t h = vsubq_s16(left, right);
    int16x8_t v = vsubq_s16(above, below);

    h = vshlq_n_s16(h, 7);
    v = vshlq_n_s16(v, 7);

    h = mulhi_s16(h, depth);
    v = mulhi_s16(v, depth);

    v = vmaxq_s16(v, y_limit_min);
    v = vminq_s16(v, y_limit_max);

    int16x8_t remainder_h = h;
    int16x8_t remainder_v = v;

    if (SMAGL) {
        // Register shift: a shift immediate of 0 is not allowed, and this compiles for SMAGL=0 too.
        remainder_h = vshlq_s16(remainder_h, vdupq_n_s16(SMAGL));
        remainder_v = vshlq_s16(remainder_v, vdupq_n_s16(SMAGL));
    }

    remainder_h = vandq_s16(remainder_h, word_127);
    remainder_v = vandq_s16(remainder_v, word_127);

    h = vshrq_n_s16(h, 7 - SMAGL);
    v = vshrq_n_s16(v, 7 - SMAGL);

    int16x8_t xx = vdupq_n_s16(sat16(x << SMAGL));

    h = vqaddq_s16(h, xx);

    // remainder_h is only kept where x_limit_min <= h < x_limit_max.
    remainder_h = vandq_s16(remainder_h, vreinterpretq_s16_u16(vcgtq_s16(x_limit_max, h)));
    remainder_h = vbicq_s16(remainder_h, vreinterpretq_s16_u16(vcgtq_s16(x_limit_min, h)));

    h = vmaxq_s16(h, x_limit_min);
    h = vminq_s16(h, x_limit_max);

    // h and v contain the displacement now.

    // Byte offsets of the source pixels: v * src_stride + h.
    int32x4_t offset_lo = vmlaq_n_s32(vmovl_s16(vget_low_s16(h)), vmovl_s16(vget_low_s16(v)), src_stride);
    int32x4_t offset_hi = vmlaq_n_s32(vmovl_s16(vget_high_s16(h)), vmovl_s16(vget_high_s16(v)), src_stride);

    int32_t offsets[8];
    vst1q_s32(offsets, offset_lo);
    vst1q_s32(offsets + 4, offset_hi);

    // Gather the two horizontal neighbours on the two source rows for every lane.
    uint8_t left0[8], right0[8], left1[8], right1[8];
    for (int i = 0; i < 8; i++) {
        const uint8_t *p = srcp + offsets[i] + i * SMAG;
        left0[i] = p[0];
        right0[i] = p[1];
        left1[i] = p[src_stride];
        right1[i] = p[src_stride + 1];
    }

    int16x8_t l0 = load_u8_as_s16(left0);
    int16x8_t l1 = load_u8_as_s16(left1);
    int16x8_t r0 = load_u8_as_s16(right0);
    int16x8_t r1 = load_u8_as_s16(right1);

    // All products stay below 32768: pixel * weight with weights summing to 128.
    int16x8_t weight_left = vsubq_s16(word_128, remainder_h);

    l0 = vmulq_s16(l0, weight_left);
    l1 = vmulq_s16(l1, weight_left);

    r0 = vmulq_s16(r0, remainder_h);
    r1 = vmulq_s16(r1, remainder_h);

    int16x8_t line0 = vaddq_s16(l0, r0);
    int16x8_t line1 = vaddq_s16(l1, r1);

    line0 = vaddq_s16(line0, word_64);
    line1 = vaddq_s16(line1, word_64);

    line0 = vshrq_n_s16(line0, 7);
    line1 = vshrq_n_s16(line1, 7);

    line0 = vmulq_s16(line0, vsubq_s16(word_128, remainder_v));
    line1 = vmulq_s16(line1, remainder_v);

    int16x8_t result = vaddq_s16(line0, line1);

    result = vaddq_s16(result, word_64);

    result = vshrq_n_s16(result, 7);

    vst1_u8(dstp + x, vqmovun_s16(result));
}


template <int SMAGL> // 0 or 2
static void warp_u8_neon(const uint8_t *srcp, const uint8_t *edgep, uint8_t *dstp, int src_stride, int edge_stride, int dst_stride, int width, int height, int depth_scalar) {
    const int SMAG = 1 << SMAGL;

    const int16x8_t depth = vdupq_n_s16(sat16(depth_scalar << 8));

    // Per-lane limits: lane i handles pixel x + i, but h only carries x, so the limits are
    // shifted by -i * SMAG (the SSE2 code does the same).
    int16_t x_limit_min_lanes[8];
    int16_t x_limit_max_lanes[8];
    for (int i = 0; i < 8; i++) {
        x_limit_min_lanes[i] = (int16_t)(-i * SMAG);
        x_limit_max_lanes[i] = (int16_t)((width - 1 - i) * SMAG);
    }
    const int16x8_t x_limit_min = vld1q_s16(x_limit_min_lanes);
    const int16x8_t x_limit_max = vld1q_s16(x_limit_max_lanes);


    for (int y = 0; y < height; y++) {
        const int16x8_t y_limit_min = vdupq_n_s16(sat16(-y * 128));
        const int16x8_t y_limit_max = vdupq_n_s16(sat16((height - y) * 128 - 129)); // (height - y - 1) * 128 - 1

        warp_edge_c<SMAGL>(srcp, edgep, dstp, src_stride, edge_stride, width, height, 0, y, depth_scalar);

        // Full blocks of 8 pixels inside [1, width - 2], then one more block ending at width - 2
        // if pixels are left (a narrow row still starts at 1). Nothing is written past the row.
        int x = 1;
        for ( ; x <= width - 9; x += 8)
            warp_vector_u8_neon<SMAGL>(srcp, edgep, dstp, src_stride, edge_stride, height, x, y, depth, x_limit_min, x_limit_max, y_limit_min, y_limit_max);
        if (x < width - 1) {
            x = std::max(width - 9, 1);
            warp_vector_u8_neon<SMAGL>(srcp, edgep, dstp, src_stride, edge_stride, height, x, y, depth, x_limit_min, x_limit_max, y_limit_min, y_limit_max);
        }

        warp_edge_c<SMAGL>(srcp, edgep, dstp, src_stride, edge_stride, width, height, width - 1, y, depth_scalar);

        srcp += src_stride * SMAG;
        edgep += edge_stride;
        dstp += dst_stride;
    }
}


void sobel_u8_neon(const uint8_t *srcp, uint8_t *dstp, int stride, int width, int height, int thresh, int bits_per_sample) {
    sobel_neon<uint8_t>(srcp, dstp, stride, width, height, thresh, bits_per_sample);
}


void sobel_u16_neon(const uint8_t *srcp, uint8_t *dstp, int stride, int width, int height, int thresh, int bits_per_sample) {
    sobel_neon<uint16_t>(srcp, dstp, stride, width, height, thresh, bits_per_sample);
}


void blur_r6_u8_neon(uint8_t *mask, uint8_t *temp, int stride, int width, int height) {
    blur_r6_neon<uint8_t>(mask, temp, stride, width, height);
}


void blur_r6_u16_neon(uint8_t *mask, uint8_t *temp, int stride, int width, int height) {
    blur_r6_neon<uint16_t>(mask, temp, stride, width, height);
}


void blur_r2_u8_neon(uint8_t *mask, uint8_t *temp, int stride, int width, int height) {
    blur_r2_neon<uint8_t>(mask, temp, stride, width, height);
}


void blur_r2_u16_neon(uint8_t *mask, uint8_t *temp, int stride, int width, int height) {
    blur_r2_neon<uint16_t>(mask, temp, stride, width, height);
}


void warp0_u8_neon(const uint8_t *srcp, const uint8_t *edgep, uint8_t *dstp, int src_stride, int edge_stride, int dst_stride, int width, int height, int depth, int bits_per_sample) {
    (void)bits_per_sample;

    warp_u8_neon<0>(srcp, edgep, dstp, src_stride, edge_stride, dst_stride, width, height, depth);
}


void warp2_u8_neon(const uint8_t *srcp, const uint8_t *edgep, uint8_t *dstp, int src_stride, int edge_stride, int dst_stride, int width, int height, int depth, int bits_per_sample) {
    (void)bits_per_sample;

    warp_u8_neon<2>(srcp, edgep, dstp, src_stride, edge_stride, dst_stride, width, height, depth);
}
