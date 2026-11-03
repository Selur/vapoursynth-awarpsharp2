// AVX2 versions of the SSE2 kernels in sse2.cpp (same algorithm, 32 instead of 16 bytes per
// step), plus a 16 bit warp, which has no SSE2 version. Every function gives exactly the same
// result as the C code.
//
// This file is compiled with -mavx2 (/arch:AVX2) and is only called after a runtime check (see
// selectFunctions), so it must not use inline functions from the standard library: the linker
// could pick an AVX2 instance of one for the code that runs on every CPU.

#include <cstdint>
#include <cstring>
#include <immintrin.h>


#ifdef _MSC_VER
#define FORCE_INLINE __forceinline
#else
#define FORCE_INLINE inline __attribute__((always_inline))
#endif


// Planes narrower than this many vectors use the SSE2 (8 and 16 bit) or C (16 bit warp) code.
#define MIN_VECTORS 2


extern void sobel_u8_sse2(const uint8_t *srcp, uint8_t *dstp, int stride, int width, int height, int thresh, int bits_per_sample);
extern void sobel_u16_sse2(const uint8_t *srcp, uint8_t *dstp, int stride, int width, int height, int thresh, int bits_per_sample);
extern void blur_r6_u8_sse2(uint8_t *mask, uint8_t *temp, int stride, int width, int height);
extern void blur_r6_u16_sse2(uint8_t *mask, uint8_t *temp, int stride, int width, int height);
extern void blur_r2_u8_sse2(uint8_t *mask, uint8_t *temp, int stride, int width, int height);
extern void blur_r2_u16_sse2(uint8_t *mask, uint8_t *temp, int stride, int width, int height);
extern void warp0_u8_sse2(const uint8_t *srcp, const uint8_t *edgep, uint8_t *dstp, int src_stride, int edge_stride, int dst_stride, int width, int height, int depth, int bits_per_sample);
extern void warp2_u8_sse2(const uint8_t *srcp, const uint8_t *edgep, uint8_t *dstp, int src_stride, int edge_stride, int dst_stride, int width, int height, int depth, int bits_per_sample);
extern void warp0_u16_c(const uint8_t *srcp, const uint8_t *edgep, uint8_t *dstp, int src_stride, int edge_stride, int dst_stride, int width, int height, int depth, int bits_per_sample);
extern void warp2_u16_c(const uint8_t *srcp, const uint8_t *edgep, uint8_t *dstp, int src_stride, int edge_stride, int dst_stride, int width, int height, int depth, int bits_per_sample);


static FORCE_INLINE int min_int(int a, int b) {
    return a < b ? a : b;
}


static FORCE_INLINE int max_int(int a, int b) {
    return a > b ? a : b;
}


// What _mm_packs_epi32 does to one value.
static FORCE_INLINE int16_t saturate_int16(int a) {
    return (int16_t)min_int(max_int(a, -32768), 32767);
}


template <typename PixelType>
static FORCE_INLINE __m256i mm256_avg_epu(const __m256i &a, const __m256i &b) {
    if (sizeof(PixelType) == 1)
        return _mm256_avg_epu8(a, b);
    else
        return _mm256_avg_epu16(a, b);
}


template <typename PixelType>
static FORCE_INLINE __m256i mm256_subs_epu(const __m256i &a, const __m256i &b) {
    if (sizeof(PixelType) == 1)
        return _mm256_subs_epu8(a, b);
    else
        return _mm256_subs_epu16(a, b);
}


template <typename PixelType>
static FORCE_INLINE __m256i mm256_adds_epu(const __m256i &a, const __m256i &b) {
    if (sizeof(PixelType) == 1)
        return _mm256_adds_epu8(a, b);
    else
        return _mm256_adds_epu16(a, b);
}


template <typename PixelType>
static FORCE_INLINE __m256i mm256_max_epu(const __m256i &a, const __m256i &b) {
    if (sizeof(PixelType) == 1)
        return _mm256_max_epu8(a, b);
    else
        return _mm256_max_epu16(a, b);
}


template <typename PixelType>
static FORCE_INLINE __m256i mm256_min_epu(const __m256i &a, const __m256i &b) {
    if (sizeof(PixelType) == 1)
        return _mm256_min_epu8(a, b);
    else
        return _mm256_min_epu16(a, b);
}


template <typename PixelType>
static FORCE_INLINE __m256i mm256_set1_epi(PixelType a) {
    if (sizeof(PixelType) == 1)
        return _mm256_set1_epi8((char)a);
    else
        return _mm256_set1_epi16((short)a);
}


static FORCE_INLINE __m256i loadu(const void *p) {
    return _mm256_loadu_si256((const __m256i *)p);
}


static FORCE_INLINE void storeu(void *p, const __m256i &a) {
    _mm256_storeu_si256((__m256i *)p, a);
}


template <typename PixelType>
static FORCE_INLINE void sobel_ymmword_avx2(const PixelType *srcp, PixelType *dstp, int stride, const __m256i &th, const __m256i &pixel_max) {
    __m256i a11, a21, a31,
            a12,      a32,
            a13, a23, a33;

    a11 = loadu(srcp - stride - 1);
    a21 = loadu(srcp - stride);
    a31 = loadu(srcp - stride + 1);

    a12 = loadu(srcp - 1);
    a32 = loadu(srcp + 1);

    a13 = loadu(srcp + stride - 1);
    a23 = loadu(srcp + stride);
    a33 = loadu(srcp + stride + 1);

    __m256i avg_up    = mm256_avg_epu<PixelType>(a21, mm256_avg_epu<PixelType>(a11, a31));
    __m256i avg_down  = mm256_avg_epu<PixelType>(a23, mm256_avg_epu<PixelType>(a13, a33));
    __m256i avg_left  = mm256_avg_epu<PixelType>(a12, mm256_avg_epu<PixelType>(a13, a11));
    __m256i avg_right = mm256_avg_epu<PixelType>(a32, mm256_avg_epu<PixelType>(a33, a31));

    __m256i abs_v = _mm256_or_si256(mm256_subs_epu<PixelType>(avg_up, avg_down), mm256_subs_epu<PixelType>(avg_down, avg_up));
    __m256i abs_h = _mm256_or_si256(mm256_subs_epu<PixelType>(avg_left, avg_right), mm256_subs_epu<PixelType>(avg_right, avg_left));

    __m256i absolute = mm256_adds_epu<PixelType>(abs_v, abs_h);
    if (sizeof(PixelType) == 2)
        absolute = mm256_min_epu<PixelType>(absolute, pixel_max);

    __m256i abs_max = mm256_max_epu<PixelType>(abs_h, abs_v);

    absolute = mm256_adds_epu<PixelType>(absolute, abs_max);
    if (sizeof(PixelType) == 2)
        absolute = mm256_min_epu<PixelType>(absolute, pixel_max);

    __m256i absolute2 = mm256_adds_epu<PixelType>(absolute, absolute);
    if (sizeof(PixelType) == 2)
        absolute2 = mm256_min_epu<PixelType>(absolute2, pixel_max);

    absolute = mm256_adds_epu<PixelType>(absolute2, absolute);
    if (sizeof(PixelType) == 2)
        absolute = mm256_min_epu<PixelType>(absolute, pixel_max);

    absolute = mm256_adds_epu<PixelType>(absolute, absolute);
    if (sizeof(PixelType) == 2)
        absolute = mm256_min_epu<PixelType>(absolute, pixel_max);

    storeu(dstp, mm256_min_epu<PixelType>(absolute, th));
}


template <typename PixelType>
static void sobel_avx2(const uint8_t *srcp8, uint8_t *dstp8, int stride, int width, int height, int thresh, int bits_per_sample) {
    const PixelType *srcp = (const PixelType *)srcp8;
    PixelType *dstp = (PixelType *)dstp8;

    stride /= sizeof(PixelType);

    __m256i pixel_max = _mm256_set1_epi16((short)((1 << bits_per_sample) - 1));

    const int pixels_in_ymm = 32 / sizeof(PixelType);

    PixelType *dstp_orig = dstp;

    srcp += stride;
    dstp += stride;

    __m256i th = mm256_set1_epi<PixelType>((PixelType)thresh);

    int width_avx2 = (width & ~(pixels_in_ymm - 1)) + 2;
    if (width_avx2 > stride)
        width_avx2 -= pixels_in_ymm;

    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width_avx2 - 1; x += pixels_in_ymm)
            sobel_ymmword_avx2<PixelType>(srcp + x, dstp + x, stride, th, pixel_max);

        // Last vector, clamped like the SSE2 code (only matters for widths below MIN_VECTORS).
        if (width + 2 > width_avx2) {
            int x = max_int(width - pixels_in_ymm - 1, 1);
            sobel_ymmword_avx2<PixelType>(srcp + x, dstp + x, stride, th, pixel_max);
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
static FORCE_INLINE void blur_r6_h_left_avx2(const PixelType *srcp, PixelType *dstp) {
    __m256i avg12 = mm256_avg_epu<PixelType>(loadu(srcp + 1), loadu(srcp + 2));
    __m256i avg34 = mm256_avg_epu<PixelType>(loadu(srcp + 3), loadu(srcp + 4));
    __m256i avg56 = mm256_avg_epu<PixelType>(loadu(srcp + 5), loadu(srcp + 6));

    __m256i avg012 = mm256_avg_epu<PixelType>(loadu(srcp), avg12);
    __m256i avg3456 = mm256_avg_epu<PixelType>(avg34, avg56);
    __m256i avg0123456 = mm256_avg_epu<PixelType>(avg012, avg3456);
    __m256i avg = mm256_avg_epu<PixelType>(avg012, avg0123456);

    storeu(dstp, avg);
}


template <typename PixelType>
static FORCE_INLINE void blur_r6_h_middle_avx2(const PixelType *srcp, PixelType *dstp) {
    __m256i avg11 = mm256_avg_epu<PixelType>(loadu(srcp - 1), loadu(srcp + 1));
    __m256i avg22 = mm256_avg_epu<PixelType>(loadu(srcp - 2), loadu(srcp + 2));
    __m256i avg33 = mm256_avg_epu<PixelType>(loadu(srcp - 3), loadu(srcp + 3));
    __m256i avg44 = mm256_avg_epu<PixelType>(loadu(srcp - 4), loadu(srcp + 4));
    __m256i avg55 = mm256_avg_epu<PixelType>(loadu(srcp - 5), loadu(srcp + 5));
    __m256i avg66 = mm256_avg_epu<PixelType>(loadu(srcp - 6), loadu(srcp + 6));

    __m256i avg12 = mm256_avg_epu<PixelType>(avg11, avg22);
    __m256i avg34 = mm256_avg_epu<PixelType>(avg33, avg44);
    __m256i avg56 = mm256_avg_epu<PixelType>(avg55, avg66);
    __m256i avg012 = mm256_avg_epu<PixelType>(loadu(srcp), avg12);
    __m256i avg3456 = mm256_avg_epu<PixelType>(avg34, avg56);
    __m256i avg0123456 = mm256_avg_epu<PixelType>(avg012, avg3456);
    __m256i avg = mm256_avg_epu<PixelType>(avg012, avg0123456);

    storeu(dstp, avg);
}


template <typename PixelType>
static FORCE_INLINE void blur_r6_h_right_avx2(const PixelType *srcp, PixelType *dstp) {
    __m256i avg12 = mm256_avg_epu<PixelType>(loadu(srcp - 1), loadu(srcp - 2));
    __m256i avg34 = mm256_avg_epu<PixelType>(loadu(srcp - 3), loadu(srcp - 4));
    __m256i avg56 = mm256_avg_epu<PixelType>(loadu(srcp - 5), loadu(srcp - 6));

    __m256i avg012 = mm256_avg_epu<PixelType>(loadu(srcp), avg12);
    __m256i avg3456 = mm256_avg_epu<PixelType>(avg34, avg56);
    __m256i avg0123456 = mm256_avg_epu<PixelType>(avg012, avg3456);
    __m256i avg = mm256_avg_epu<PixelType>(avg012, avg0123456);

    // This is the right edge. Only the highest six pixels are needed, the others keep what the
    // middle part wrote.
    const __m256i high_six = sizeof(PixelType) == 1
        ? _mm256_setr_epi64x(0, 0, 0, (long long)0xffffffffffff0000ULL)
        : _mm256_setr_epi64x(0, 0, (long long)0xffffffff00000000ULL, -1);

    storeu(dstp, _mm256_blendv_epi8(loadu(dstp), avg, high_six));
}


template <typename PixelType>
static FORCE_INLINE void blur_r6_v_top_avx2(const PixelType *srcp, PixelType *dstp, int stride) {
    __m256i l0 = loadu(srcp);
    __m256i l1 = loadu(srcp + stride);
    __m256i l2 = loadu(srcp + stride * 2);
    __m256i l3 = loadu(srcp + stride * 3);
    __m256i l4 = loadu(srcp + stride * 4);
    __m256i l5 = loadu(srcp + stride * 5);
    __m256i l6 = loadu(srcp + stride * 6);

    __m256i avg12 = mm256_avg_epu<PixelType>(l1, l2);
    __m256i avg34 = mm256_avg_epu<PixelType>(l3, l4);
    __m256i avg56 = mm256_avg_epu<PixelType>(l5, l6);

    __m256i avg3456 = mm256_avg_epu<PixelType>(avg34, avg56);
    __m256i avg012 = mm256_avg_epu<PixelType>(l0, avg12);
    __m256i avg0123456 = mm256_avg_epu<PixelType>(avg012, avg3456);
    __m256i avg = mm256_avg_epu<PixelType>(avg012, avg0123456);

    storeu(dstp, avg);
}


template <typename PixelType>
static FORCE_INLINE void blur_r6_v_middle_avx2(const PixelType *srcp, PixelType *dstp, int stride) {
    __m256i m6 = loadu(srcp - stride * 6);
    __m256i m5 = loadu(srcp - stride * 5);
    __m256i m4 = loadu(srcp - stride * 4);
    __m256i m3 = loadu(srcp - stride * 3);
    __m256i m2 = loadu(srcp - stride * 2);
    __m256i m1 = loadu(srcp - stride);
    __m256i l0 = loadu(srcp);
    __m256i l1 = loadu(srcp + stride);
    __m256i l2 = loadu(srcp + stride * 2);
    __m256i l3 = loadu(srcp + stride * 3);
    __m256i l4 = loadu(srcp + stride * 4);
    __m256i l5 = loadu(srcp + stride * 5);
    __m256i l6 = loadu(srcp + stride * 6);

    __m256i avg11 = mm256_avg_epu<PixelType>(m1, l1);
    __m256i avg22 = mm256_avg_epu<PixelType>(m2, l2);
    __m256i avg33 = mm256_avg_epu<PixelType>(m3, l3);
    __m256i avg44 = mm256_avg_epu<PixelType>(m4, l4);
    __m256i avg55 = mm256_avg_epu<PixelType>(m5, l5);
    __m256i avg66 = mm256_avg_epu<PixelType>(m6, l6);

    __m256i avg12 = mm256_avg_epu<PixelType>(avg11, avg22);
    __m256i avg34 = mm256_avg_epu<PixelType>(avg33, avg44);
    __m256i avg56 = mm256_avg_epu<PixelType>(avg55, avg66);
    __m256i avg012 = mm256_avg_epu<PixelType>(l0, avg12);
    __m256i avg3456 = mm256_avg_epu<PixelType>(avg34, avg56);
    __m256i avg0123456 = mm256_avg_epu<PixelType>(avg012, avg3456);
    __m256i avg = mm256_avg_epu<PixelType>(avg012, avg0123456);

    storeu(dstp, avg);
}


template <typename PixelType>
static FORCE_INLINE void blur_r6_v_bottom_avx2(const PixelType *srcp, PixelType *dstp, int stride) {
    __m256i m6 = loadu(srcp - stride * 6);
    __m256i m5 = loadu(srcp - stride * 5);
    __m256i m4 = loadu(srcp - stride * 4);
    __m256i m3 = loadu(srcp - stride * 3);
    __m256i m2 = loadu(srcp - stride * 2);
    __m256i m1 = loadu(srcp - stride);
    __m256i l0 = loadu(srcp);

    __m256i avg12 = mm256_avg_epu<PixelType>(m1, m2);
    __m256i avg34 = mm256_avg_epu<PixelType>(m3, m4);
    __m256i avg56 = mm256_avg_epu<PixelType>(m5, m6);
    __m256i avg012 = mm256_avg_epu<PixelType>(l0, avg12);
    __m256i avg3456 = mm256_avg_epu<PixelType>(avg34, avg56);
    __m256i avg0123456 = mm256_avg_epu<PixelType>(avg012, avg3456);
    __m256i avg = mm256_avg_epu<PixelType>(avg012, avg0123456);

    storeu(dstp, avg);
}


template <typename PixelType>
static void blur_r6_avx2(uint8_t *mask8, uint8_t *temp8, int stride, int width, int height) {
    // Horizontal blur from mask to temp.
    // Vertical blur from temp back to mask.

    PixelType *mask = (PixelType *)mask8;
    PixelType *temp = (PixelType *)temp8;

    stride /= sizeof(PixelType);

    const int pixels_in_ymm = 32 / sizeof(PixelType);

    int width_avx2 = (width & ~(pixels_in_ymm - 1)) + 12;
    if (width_avx2 > stride)
        width_avx2 -= pixels_in_ymm;

    PixelType *mask_orig = mask;
    PixelType *temp_orig = temp;

    // Horizontal blur.

    for (int y = 0; y < height; y++) {
        blur_r6_h_left_avx2<PixelType>(mask, temp);

        for (int x = 6; x < width_avx2 - 6; x += pixels_in_ymm)
            blur_r6_h_middle_avx2<PixelType>(mask + x, temp + x);

        // Last middle vector, clamped to the first middle pixel like the SSE2 code.
        if (width + 12 > width_avx2) {
            int x = max_int(width - pixels_in_ymm - 6, 6);
            blur_r6_h_middle_avx2<PixelType>(mask + x, temp + x);
        }

        blur_r6_h_right_avx2<PixelType>(mask + width - pixels_in_ymm, temp + width - pixels_in_ymm);

        mask += stride;
        temp += stride;
    }


    // Vertical blur.

    width_avx2 = width & ~(pixels_in_ymm - 1);

    mask = mask_orig;
    temp = temp_orig;
    int y;

    for (y = 0; y < 6; y++) {
        for (int x = 0; x < width_avx2; x += pixels_in_ymm)
            blur_r6_v_top_avx2<PixelType>(temp + x, mask + x, stride);

        if (width > width_avx2)
            blur_r6_v_top_avx2<PixelType>(temp + width - pixels_in_ymm, mask + width - pixels_in_ymm, stride);

        mask += stride;
        temp += stride;
    }

    for ( ; y < height - 6; y++) {
        for (int x = 0; x < width_avx2; x += pixels_in_ymm)
            blur_r6_v_middle_avx2<PixelType>(temp + x, mask + x, stride);

        if (width > width_avx2)
            blur_r6_v_middle_avx2<PixelType>(temp + width - pixels_in_ymm, mask + width - pixels_in_ymm, stride);

        mask += stride;
        temp += stride;
    }

    for ( ; y < height; y++) {
        for (int x = 0; x < width_avx2; x += pixels_in_ymm)
            blur_r6_v_bottom_avx2<PixelType>(temp + x, mask + x, stride);

        if (width > width_avx2)
            blur_r6_v_bottom_avx2<PixelType>(temp + width - pixels_in_ymm, mask + width - pixels_in_ymm, stride);

        mask += stride;
        temp += stride;
    }
}


template <typename PixelType>
static FORCE_INLINE void blur_r2_h_avx2(const PixelType *srcp, PixelType *dstp) {
    __m256i avg1 = mm256_avg_epu<PixelType>(loadu(srcp - 1), loadu(srcp + 1));
    __m256i avg2 = mm256_avg_epu<PixelType>(loadu(srcp - 2), loadu(srcp + 2));
    __m256i middle = loadu(srcp);
    __m256i avg = mm256_avg_epu<PixelType>(avg2, middle);
    avg = mm256_avg_epu<PixelType>(avg, middle);
    avg = mm256_avg_epu<PixelType>(avg, avg1);

    storeu(dstp, avg);
}


template <typename PixelType>
static FORCE_INLINE void blur_r2_v_avx2(const PixelType *srcp, PixelType *dstp, int stride_p2, int stride_p1, int stride_n1, int stride_n2) {
    __m256i m2 = loadu(srcp + stride_p2);
    __m256i m1 = loadu(srcp + stride_p1);
    __m256i l0 = loadu(srcp);
    __m256i l1 = loadu(srcp + stride_n1);
    __m256i l2 = loadu(srcp + stride_n2);

    __m256i avg1 = mm256_avg_epu<PixelType>(m1, l1);
    __m256i avg2 = mm256_avg_epu<PixelType>(m2, l2);
    __m256i avg = mm256_avg_epu<PixelType>(avg2, l0);
    avg = mm256_avg_epu<PixelType>(avg, l0);
    avg = mm256_avg_epu<PixelType>(avg, avg1);

    storeu(dstp, avg);
}


template <typename PixelType>
static void blur_r2_avx2(uint8_t *mask8, uint8_t *temp8, int stride, int width, int height) {
    // Horizontal blur from mask to temp.
    // Vertical blur from temp back to mask.

    PixelType *mask = (PixelType *)mask8;
    PixelType *temp = (PixelType *)temp8;

    stride /= sizeof(PixelType);

    const int pixels_in_ymm = 32 / sizeof(PixelType);

    int width_avx2 = (width & ~(pixels_in_ymm - 1)) + 4;
    if (width_avx2 > stride)
        width_avx2 -= pixels_in_ymm;

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

        for (int x = 2; x < width_avx2 - 2; x += pixels_in_ymm)
            blur_r2_h_avx2<PixelType>(mask + x, temp + x);

        // Last middle vector, clamped to the first middle pixel (see blur_r6).
        if (width + 4 > width_avx2) {
            int x = max_int(width - pixels_in_ymm - 2, 2);
            blur_r2_h_avx2<PixelType>(mask + x, temp + x);
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

    width_avx2 = width & ~(pixels_in_ymm - 1);

    mask = mask_orig;
    temp = temp_orig;

    for (int y = 0; y < height; y++) {
        int stride_p1 = y ? -stride : 0;
        int stride_p2 = y > 1 ? stride_p1 * 2 : stride_p1;
        int stride_n1 = y < height - 1 ? stride : 0;
        int stride_n2 = y < height - 2 ? stride_n1 * 2 : stride_n1;

        for (int x = 0; x < width_avx2; x += pixels_in_ymm)
            blur_r2_v_avx2<PixelType>(temp + x, mask + x, stride_p2, stride_p1, stride_n1, stride_n2);

        if (width > width_avx2)
            blur_r2_v_avx2<PixelType>(temp + width - pixels_in_ymm, mask + width - pixels_in_ymm, stride_p2, stride_p1, stride_n1, stride_n2);

        mask += stride;
        temp += stride;
    }
}


// One pixel of warp_c, for the left and right columns.
template <int SMAGL, typename PixelType>
static FORCE_INLINE void warp_pixel_c(const PixelType *srcp, const PixelType *edgep, PixelType *dstp, int src_stride, int edge_stride, int width, int height, int x, int y, int depth, int bits_per_sample) {
    int SMAG = 1 << SMAGL;

    int extra_bits = bits_per_sample - 8;
    int pixel_max = (1 << bits_per_sample) - 1;

    depth <<= 8;

    const int x_limit_min = 0 * SMAG;
    const int x_limit_max = (width - 1) * SMAG;

    int y_limit_min = -y * 128;
    int y_limit_max = (height - y) * 128 - 129; // (height - y - 1) * 128 - 1

    // calculate displacement

    int above = edgep[x - (y ? edge_stride : 0)];
    int below = edgep[x + (y < height - 1 ? edge_stride : 0)];

    int left = edgep[x - (x ? 1 : 0)];
    int right = edgep[x + (x < width - 1 ? 1 : 0)];

    int h = left - right;
    int v = above - below;

    if (sizeof(PixelType) == 2) {
        h >>= extra_bits;
        v >>= extra_bits;
    }

    h <<= 7;
    v <<= 7;

    h *= depth;
    h >>= 16;
    v *= depth;
    v >>= 16;

    v = max_int(v, y_limit_min);
    v = min_int(v, y_limit_max);

    int remainder_h = h;
    int remainder_v = v;

    if (SMAGL) {
        remainder_h <<= SMAGL;
        remainder_v <<= SMAGL;
    }

    remainder_h &= 127;
    remainder_v &= 127;

    h >>= 7 - SMAGL;
    v >>= 7 - SMAGL;

    h += x << SMAGL;
    h = min_int(max_int(h, -32768), 32767); // likely pointless

    bool remainder_needed = (x_limit_max > h) && !(x_limit_min > h);
    if (!remainder_needed)
        remainder_h = 0; // probably correct

    h = min_int(h, x_limit_max);
    h = max_int(h, x_limit_min);

    // h and v contain the displacement now.

    int s00 = srcp[v * src_stride + h];
    int s01 = srcp[v * src_stride + h + 1];
    int s10 = srcp[(v + 1) * src_stride + h];
    int s11 = srcp[(v + 1) * src_stride + h + 1];

    int s0 = s00 * (128 - remainder_h);
    int s1 = s10 * (128 - remainder_h);

    s0 += s01 * remainder_h;
    s1 += s11 * remainder_h;

    s0 += 64;
    s1 += 64;

    s0 >>= 7;
    s1 >>= 7;

    s0 *= 128 - remainder_v;
    s1 *= remainder_v;

    int s = s0 + s1;

    s += 64;

    s >>= 7;

    dstp[x] = min_int(max_int(s, 0), pixel_max);
}


// 16 pixels of warp_mmword_u8_sse2. The SSE2 code reads the two source pixels it interpolates
// with one 16 bit load per pixel and row; here a 32 bit gather reads them, for the upper row
// starting at the pair (low half) and for the lower row ending with it (high half), so that no
// gather reads before the first or after the last row the SSE2 code reads.
template <int SMAGL>
static FORCE_INLINE void warp_ymmword_u8_avx2(const uint8_t *srcp, const uint8_t *edgep, uint8_t *dstp, int src_stride, int edge_stride, int height, int x, int y, const __m256i &depth, const __m256i &x_limit_min, const __m256i &x_limit_max, const __m256i &y_limit_min, const __m256i &y_limit_max, const __m256i &lane_offsets_lo, const __m256i &lane_offsets_hi, const __m256i &stride32) {
    const __m256i word_64 = _mm256_set1_epi16(64);
    const __m256i word_127 = _mm256_set1_epi16(127);
    const __m256i word_128 = _mm256_set1_epi16(128);
    const __m256i word_255 = _mm256_set1_epi16(255);
    const __m256i dword_65535 = _mm256_set1_epi32(65535);

    // calculate displacement

    __m256i above = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(edgep + x - (y ? edge_stride : 0))));
    __m256i below = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(edgep + x + (y < height - 1 ? edge_stride : 0))));

    __m256i left = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(edgep + x - 1)));
    __m256i right = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(edgep + x + 1)));

    __m256i h = _mm256_sub_epi16(left, right);
    __m256i v = _mm256_sub_epi16(above, below);

    h = _mm256_slli_epi16(h, 7);
    v = _mm256_slli_epi16(v, 7);

    h = _mm256_mulhi_epi16(h, depth);
    v = _mm256_mulhi_epi16(v, depth);

    v = _mm256_max_epi16(v, y_limit_min);
    v = _mm256_min_epi16(v, y_limit_max);

    __m256i remainder_h = h;
    __m256i remainder_v = v;

    if (SMAGL) {
        remainder_h = _mm256_slli_epi16(remainder_h, SMAGL);
        remainder_v = _mm256_slli_epi16(remainder_v, SMAGL);
    }

    remainder_h = _mm256_and_si256(remainder_h, word_127);
    remainder_v = _mm256_and_si256(remainder_v, word_127);

    h = _mm256_srai_epi16(h, 7 - SMAGL);
    v = _mm256_srai_epi16(v, 7 - SMAGL);

    __m256i xx = _mm256_set1_epi16(saturate_int16(x << SMAGL));

    h = _mm256_adds_epi16(h, xx);

    remainder_h = _mm256_and_si256(remainder_h, _mm256_cmpgt_epi16(x_limit_max, h));
    remainder_h = _mm256_andnot_si256(_mm256_cmpgt_epi16(x_limit_min, h), remainder_h);

    h = _mm256_max_epi16(h, x_limit_min);
    h = _mm256_min_epi16(h, x_limit_max);

    // h and v contain the displacement now.

    __m256i offset_lo = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(v)), stride32),
                                         _mm256_cvtepi16_epi32(_mm256_castsi256_si128(h)));
    __m256i offset_hi = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(v, 1)), stride32),
                                         _mm256_cvtepi16_epi32(_mm256_extracti128_si256(h, 1)));
    offset_lo = _mm256_add_epi32(offset_lo, lane_offsets_lo);
    offset_hi = _mm256_add_epi32(offset_hi, lane_offsets_hi);

    const int *row0 = (const int *)(srcp);
    const int *row1 = (const int *)(srcp + src_stride - 2);

    __m256i line0_lo = _mm256_and_si256(_mm256_i32gather_epi32(row0, offset_lo, 1), dword_65535);
    __m256i line0_hi = _mm256_and_si256(_mm256_i32gather_epi32(row0, offset_hi, 1), dword_65535);
    __m256i line1_lo = _mm256_srli_epi32(_mm256_i32gather_epi32(row1, offset_lo, 1), 16);
    __m256i line1_hi = _mm256_srli_epi32(_mm256_i32gather_epi32(row1, offset_hi, 1), 16);

    __m256i line0 = _mm256_permute4x64_epi64(_mm256_packus_epi32(line0_lo, line0_hi), 0xD8);
    __m256i line1 = _mm256_permute4x64_epi64(_mm256_packus_epi32(line1_lo, line1_hi), 0xD8);

    __m256i left0 = _mm256_and_si256(line0, word_255);
    __m256i left1 = _mm256_and_si256(line1, word_255);

    __m256i right0 = _mm256_srli_epi16(line0, 8);
    __m256i right1 = _mm256_srli_epi16(line1, 8);

    left0 = _mm256_mullo_epi16(left0, _mm256_sub_epi16(word_128, remainder_h));
    left1 = _mm256_mullo_epi16(left1, _mm256_sub_epi16(word_128, remainder_h));

    right0 = _mm256_mullo_epi16(right0, remainder_h);
    right1 = _mm256_mullo_epi16(right1, remainder_h);

    line0 = _mm256_add_epi16(left0, right0);
    line1 = _mm256_add_epi16(left1, right1);

    line0 = _mm256_add_epi16(line0, word_64);
    line1 = _mm256_add_epi16(line1, word_64);

    line0 = _mm256_srai_epi16(line0, 7);
    line1 = _mm256_srai_epi16(line1, 7);

    line0 = _mm256_mullo_epi16(line0, _mm256_sub_epi16(word_128, remainder_v));
    line1 = _mm256_mullo_epi16(line1, remainder_v);

    __m256i result = _mm256_add_epi16(line0, line1);

    result = _mm256_add_epi16(result, word_64);

    result = _mm256_srai_epi16(result, 7);

    result = _mm256_permute4x64_epi64(_mm256_packus_epi16(result, result), 0xD8);

    _mm_storeu_si128((__m128i *)(dstp + x), _mm256_castsi256_si128(result));
}


template <int SMAGL> // 0 or 2
static void warp_u8_avx2(const uint8_t *srcp, const uint8_t *edgep, uint8_t *dstp, int src_stride, int edge_stride, int dst_stride, int width, int height, int depth_scalar) {
    int SMAG = 1 << SMAGL;

    __m256i depth = _mm256_set1_epi16(saturate_int16(depth_scalar << 8));

    __m256i x_limit_min, x_limit_max;
    {
        alignas(32) int16_t lmin[16], lmax[16];
        for (int i = 0; i < 16; i++) {
            lmin[i] = (int16_t)(-i * SMAG);
            lmax[i] = (int16_t)((width - 1 - i) * SMAG);
        }
        x_limit_min = _mm256_load_si256((const __m256i *)lmin);
        x_limit_max = _mm256_load_si256((const __m256i *)lmax);
    }

    const __m256i lane_offsets_lo = _mm256_setr_epi32(0 * SMAG, 1 * SMAG, 2 * SMAG, 3 * SMAG, 4 * SMAG, 5 * SMAG, 6 * SMAG, 7 * SMAG);
    const __m256i lane_offsets_hi = _mm256_add_epi32(lane_offsets_lo, _mm256_set1_epi32(8 * SMAG));

    const __m256i stride32 = _mm256_set1_epi32(src_stride);

    for (int y = 0; y < height; y++) {
        __m256i y_limit_min = _mm256_set1_epi16(saturate_int16(-y * 128));
        __m256i y_limit_max = _mm256_set1_epi16(saturate_int16((height - y) * 128 - 129)); // (height - y - 1) * 128 - 1

        warp_pixel_c<SMAGL, uint8_t>(srcp, edgep, dstp, src_stride, edge_stride, width, height, 0, y, depth_scalar, 8);

        // Full blocks of 16 pixels inside [1, width - 2], then one more block ending at width - 2
        // if pixels are left, like the SSE2 code. Nothing is written past the row.
        int x = 1;
        for ( ; x <= width - 17; x += 16)
            warp_ymmword_u8_avx2<SMAGL>(srcp, edgep, dstp, src_stride, edge_stride, height, x, y, depth, x_limit_min, x_limit_max, y_limit_min, y_limit_max, lane_offsets_lo, lane_offsets_hi, stride32);

        if (x < width - 1)
            warp_ymmword_u8_avx2<SMAGL>(srcp, edgep, dstp, src_stride, edge_stride, height, max_int(width - 17, 1), y, depth, x_limit_min, x_limit_max, y_limit_min, y_limit_max, lane_offsets_lo, lane_offsets_hi, stride32);

        warp_pixel_c<SMAGL, uint8_t>(srcp, edgep, dstp, src_stride, edge_stride, width, height, width - 1, y, depth_scalar, 8);

        srcp += src_stride * SMAG;
        edgep += edge_stride;
        dstp += dst_stride;
    }
}


// 8 pixels of warp_c<SMAGL, uint16_t>, in 32 bit lanes like the C code. A 32 bit gather reads the
// two source pixels that are interpolated horizontally.
template <int SMAGL>
static FORCE_INLINE void warp_ymmword_u16_avx2(const uint16_t *srcp, const uint16_t *edgep, uint16_t *dstp, int src_stride, int edge_stride, int height, int x, int y, int extra_bits, const __m256i &depth, const __m256i &x_limit_max, const __m256i &y_limit_min, const __m256i &y_limit_max, const __m256i &lane_x, const __m256i &stride32, const __m256i &pixel_max) {
    const __m256i dword_64 = _mm256_set1_epi32(64);
    const __m256i dword_127 = _mm256_set1_epi32(127);
    const __m256i dword_128 = _mm256_set1_epi32(128);
    const __m256i dword_65535 = _mm256_set1_epi32(65535);
    const __m256i zero = _mm256_setzero_si256();

    // calculate displacement

    __m256i above = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(edgep + x - (y ? edge_stride : 0))));
    __m256i below = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(edgep + x + (y < height - 1 ? edge_stride : 0))));

    __m256i left = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(edgep + x - 1)));
    __m256i right = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(edgep + x + 1)));

    __m128i shift = _mm_cvtsi32_si128(extra_bits);

    __m256i h = _mm256_sra_epi32(_mm256_sub_epi32(left, right), shift);
    __m256i v = _mm256_sra_epi32(_mm256_sub_epi32(above, below), shift);

    h = _mm256_slli_epi32(h, 7);
    v = _mm256_slli_epi32(v, 7);

    h = _mm256_srai_epi32(_mm256_mullo_epi32(h, depth), 16);
    v = _mm256_srai_epi32(_mm256_mullo_epi32(v, depth), 16);

    v = _mm256_max_epi32(v, y_limit_min);
    v = _mm256_min_epi32(v, y_limit_max);

    __m256i remainder_h = h;
    __m256i remainder_v = v;

    if (SMAGL) {
        remainder_h = _mm256_slli_epi32(remainder_h, SMAGL);
        remainder_v = _mm256_slli_epi32(remainder_v, SMAGL);
    }

    remainder_h = _mm256_and_si256(remainder_h, dword_127);
    remainder_v = _mm256_and_si256(remainder_v, dword_127);

    h = _mm256_srai_epi32(h, 7 - SMAGL);
    v = _mm256_srai_epi32(v, 7 - SMAGL);

    h = _mm256_add_epi32(h, _mm256_add_epi32(lane_x, _mm256_set1_epi32(x << SMAGL)));
    h = _mm256_min_epi32(_mm256_max_epi32(h, _mm256_set1_epi32(-32768)), _mm256_set1_epi32(32767)); // likely pointless

    remainder_h = _mm256_and_si256(remainder_h, _mm256_cmpgt_epi32(x_limit_max, h));
    remainder_h = _mm256_andnot_si256(_mm256_cmpgt_epi32(zero, h), remainder_h);

    h = _mm256_min_epi32(h, x_limit_max);
    h = _mm256_max_epi32(h, zero);

    // h and v contain the displacement now.

    __m256i offset = _mm256_add_epi32(_mm256_mullo_epi32(v, stride32), h);

    __m256i line0 = _mm256_i32gather_epi32((const int *)srcp, offset, 2);
    __m256i line1 = _mm256_i32gather_epi32((const int *)(srcp + src_stride), offset, 2);

    __m256i inv_remainder_h = _mm256_sub_epi32(dword_128, remainder_h);

    __m256i s0 = _mm256_mullo_epi32(_mm256_and_si256(line0, dword_65535), inv_remainder_h);
    __m256i s1 = _mm256_mullo_epi32(_mm256_and_si256(line1, dword_65535), inv_remainder_h);

    s0 = _mm256_add_epi32(s0, _mm256_mullo_epi32(_mm256_srli_epi32(line0, 16), remainder_h));
    s1 = _mm256_add_epi32(s1, _mm256_mullo_epi32(_mm256_srli_epi32(line1, 16), remainder_h));

    s0 = _mm256_srai_epi32(_mm256_add_epi32(s0, dword_64), 7);
    s1 = _mm256_srai_epi32(_mm256_add_epi32(s1, dword_64), 7);

    s0 = _mm256_mullo_epi32(s0, _mm256_sub_epi32(dword_128, remainder_v));
    s1 = _mm256_mullo_epi32(s1, remainder_v);

    __m256i s = _mm256_srai_epi32(_mm256_add_epi32(_mm256_add_epi32(s0, s1), dword_64), 7);

    s = _mm256_min_epi32(_mm256_max_epi32(s, zero), pixel_max);

    s = _mm256_permute4x64_epi64(_mm256_packus_epi32(s, s), 0xD8);

    _mm_storeu_si128((__m128i *)(dstp + x), _mm256_castsi256_si128(s));
}


template <int SMAGL> // 0 or 2
static void warp_u16_avx2(const uint8_t *srcp8, const uint8_t *edgep8, uint8_t *dstp8, int src_stride, int edge_stride, int dst_stride, int width, int height, int depth_scalar, int bits_per_sample) {
    const uint16_t *srcp = (const uint16_t *)srcp8;
    const uint16_t *edgep = (const uint16_t *)edgep8;
    uint16_t *dstp = (uint16_t *)dstp8;

    src_stride /= 2;
    edge_stride /= 2;
    dst_stride /= 2;

    int SMAG = 1 << SMAGL;

    int extra_bits = bits_per_sample - 8;

    const __m256i depth = _mm256_set1_epi32(depth_scalar << 8);
    const __m256i x_limit_max = _mm256_set1_epi32((width - 1) * SMAG);
    const __m256i lane_x = _mm256_setr_epi32(0 * SMAG, 1 * SMAG, 2 * SMAG, 3 * SMAG, 4 * SMAG, 5 * SMAG, 6 * SMAG, 7 * SMAG);
    const __m256i stride32 = _mm256_set1_epi32(src_stride);
    const __m256i pixel_max = _mm256_set1_epi32((1 << bits_per_sample) - 1);

    for (int y = 0; y < height; y++) {
        __m256i y_limit_min = _mm256_set1_epi32(-y * 128);
        __m256i y_limit_max = _mm256_set1_epi32((height - y) * 128 - 129); // (height - y - 1) * 128 - 1

        warp_pixel_c<SMAGL, uint16_t>(srcp, edgep, dstp, src_stride, edge_stride, width, height, 0, y, depth_scalar, bits_per_sample);

        // The left and right neighbours of every pixel must be inside the row.
        int x;
        for (x = 1; x + 8 < width; x += 8)
            warp_ymmword_u16_avx2<SMAGL>(srcp, edgep, dstp, src_stride, edge_stride, height, x, y, extra_bits, depth, x_limit_max, y_limit_min, y_limit_max, lane_x, stride32, pixel_max);

        if (x < width - 1)
            warp_ymmword_u16_avx2<SMAGL>(srcp, edgep, dstp, src_stride, edge_stride, height, width - 9, y, extra_bits, depth, x_limit_max, y_limit_min, y_limit_max, lane_x, stride32, pixel_max);

        warp_pixel_c<SMAGL, uint16_t>(srcp, edgep, dstp, src_stride, edge_stride, width, height, width - 1, y, depth_scalar, bits_per_sample);

        srcp += src_stride * SMAG;
        edgep += edge_stride;
        dstp += dst_stride;
    }
}


void sobel_u8_avx2(const uint8_t *srcp, uint8_t *dstp, int stride, int width, int height, int thresh, int bits_per_sample) {
    if (width < 32 * MIN_VECTORS)
        sobel_u8_sse2(srcp, dstp, stride, width, height, thresh, bits_per_sample);
    else
        sobel_avx2<uint8_t>(srcp, dstp, stride, width, height, thresh, bits_per_sample);
}


void sobel_u16_avx2(const uint8_t *srcp, uint8_t *dstp, int stride, int width, int height, int thresh, int bits_per_sample) {
    if (width < 16 * MIN_VECTORS)
        sobel_u16_sse2(srcp, dstp, stride, width, height, thresh, bits_per_sample);
    else
        sobel_avx2<uint16_t>(srcp, dstp, stride, width, height, thresh, bits_per_sample);
}


void blur_r6_u8_avx2(uint8_t *mask, uint8_t *temp, int stride, int width, int height) {
    if (width < 32 * MIN_VECTORS)
        blur_r6_u8_sse2(mask, temp, stride, width, height);
    else
        blur_r6_avx2<uint8_t>(mask, temp, stride, width, height);
}


void blur_r6_u16_avx2(uint8_t *mask, uint8_t *temp, int stride, int width, int height) {
    if (width < 16 * MIN_VECTORS)
        blur_r6_u16_sse2(mask, temp, stride, width, height);
    else
        blur_r6_avx2<uint16_t>(mask, temp, stride, width, height);
}


void blur_r2_u8_avx2(uint8_t *mask, uint8_t *temp, int stride, int width, int height) {
    if (width < 32 * MIN_VECTORS)
        blur_r2_u8_sse2(mask, temp, stride, width, height);
    else
        blur_r2_avx2<uint8_t>(mask, temp, stride, width, height);
}


void blur_r2_u16_avx2(uint8_t *mask, uint8_t *temp, int stride, int width, int height) {
    if (width < 16 * MIN_VECTORS)
        blur_r2_u16_sse2(mask, temp, stride, width, height);
    else
        blur_r2_avx2<uint16_t>(mask, temp, stride, width, height);
}


void warp0_u8_avx2(const uint8_t *srcp, const uint8_t *edgep, uint8_t *dstp, int src_stride, int edge_stride, int dst_stride, int width, int height, int depth, int bits_per_sample) {
    if (width < 16 * MIN_VECTORS)
        warp0_u8_sse2(srcp, edgep, dstp, src_stride, edge_stride, dst_stride, width, height, depth, bits_per_sample);
    else
        warp_u8_avx2<0>(srcp, edgep, dstp, src_stride, edge_stride, dst_stride, width, height, depth);
}


void warp2_u8_avx2(const uint8_t *srcp, const uint8_t *edgep, uint8_t *dstp, int src_stride, int edge_stride, int dst_stride, int width, int height, int depth, int bits_per_sample) {
    if (width < 16 * MIN_VECTORS)
        warp2_u8_sse2(srcp, edgep, dstp, src_stride, edge_stride, dst_stride, width, height, depth, bits_per_sample);
    else
        warp_u8_avx2<2>(srcp, edgep, dstp, src_stride, edge_stride, dst_stride, width, height, depth);
}


void warp0_u16_avx2(const uint8_t *srcp, const uint8_t *edgep, uint8_t *dstp, int src_stride, int edge_stride, int dst_stride, int width, int height, int depth, int bits_per_sample) {
    if (width < 8 * MIN_VECTORS)
        warp0_u16_c(srcp, edgep, dstp, src_stride, edge_stride, dst_stride, width, height, depth, bits_per_sample);
    else
        warp_u16_avx2<0>(srcp, edgep, dstp, src_stride, edge_stride, dst_stride, width, height, depth, bits_per_sample);
}


void warp2_u16_avx2(const uint8_t *srcp, const uint8_t *edgep, uint8_t *dstp, int src_stride, int edge_stride, int dst_stride, int width, int height, int depth, int bits_per_sample) {
    if (width < 8 * MIN_VECTORS)
        warp2_u16_c(srcp, edgep, dstp, src_stride, edge_stride, dst_stride, width, height, depth, bits_per_sample);
    else
        warp_u16_avx2<2>(srcp, edgep, dstp, src_stride, edge_stride, dst_stride, width, height, depth, bits_per_sample);
}
