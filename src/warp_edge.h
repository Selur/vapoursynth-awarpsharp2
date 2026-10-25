#ifndef AWARPSHARP2_WARP_EDGE_H
#define AWARPSHARP2_WARP_EDGE_H

#include <algorithm>
#include <cstdint>


// Scalar warp of one 8 bit pixel, used by the SIMD code (SSE2, NEON) for the first and the
// last pixel of every row, where the vectorised code would read outside the mask.
template <int SMAGL>
static inline void warp_edge_c(const uint8_t *srcp, const uint8_t *edgep, uint8_t *dstp, int src_stride, int edge_stride, int width, int height, int x, int y, int depth) {
    int SMAG = 1 << SMAGL;

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

    h <<= 7;
    v <<= 7;

    h *= depth;
    h >>= 16;
    v *= depth;
    v >>= 16;

    v = std::max(v, y_limit_min);
    v = std::min(v, y_limit_max);

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
    h = std::min(std::max(h, -32768), 32767); // likely pointless

    bool remainder_needed = (x_limit_max > h) && !(x_limit_min > h);
    if (!remainder_needed)
        remainder_h = 0; // probably correct

    h = std::min(h, x_limit_max);
    h = std::max(h, x_limit_min);

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

    dstp[x] = std::min(std::max(s, 0), 255);
}

#endif // AWARPSHARP2_WARP_EDGE_H
