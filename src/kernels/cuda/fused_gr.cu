// src/kernels/cuda/fused_gr.cu - see include/strata/kernels/fused_gr.hpp.
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <string>

namespace strata::kernels {
namespace {

constexpr int N = 2560;         // n_embd
constexpr int HC = 4;           // streams
constexpr int D = N * HC;       // 10240
constexpr int LR = 320;         // hc_lr
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr int DOWN_BLOCKS = LR / WARPS;          // 40 blocks of 8 rows; one more for the inject rows
constexpr int UP_COLS = 32;                      // columns d per `up` block (x 4 streams = 128 rows)
constexpr int UP_BLOCKS = N / UP_COLS;           // 80

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + __expf(-x)); }

// 8 bf16 packed in a uint4 against 8 floats.
__device__ __forceinline__ float dot8(const uint4 w, const float* x) {
    float acc = 0.0f;
    const uint32_t v[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        acc = fmaf(__uint_as_float(v[j] << 16), x[2 * j], acc);
        acc = fmaf(__uint_as_float(v[j] & 0xffff0000u), x[2 * j + 1], acc);
    }
    return acc;
}

__global__ void __launch_bounds__(THREADS) gr_down_kernel(FusedGrArgs a) {
    __shared__ __align__(16) float xn[D];
    __shared__ float part[WARPS][HC];
    __shared__ float s_rs[HC];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    float gw[HC];
#pragma unroll
    for (int c = 0; c < HC; ++c) gw[c] = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
    // 1. R' * w_norm into shared memory, and the per-stream sums of squares of R'.
    float ss[HC] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = t * 4; i < D; i += THREADS * 4) {
        const int c = i / N, d = i - c * N;
        float4 r = *reinterpret_cast<const float4*>(a.R + i);
        if (a.apply) {
            const float4 b = *reinterpret_cast<const float4*>(a.bo_prev + d);
            r.x = fmaf(b.x, gw[c], r.x); r.y = fmaf(b.y, gw[c], r.y);
            r.z = fmaf(b.z, gw[c], r.z); r.w = fmaf(b.w, gw[c], r.w);
        }
        const float4 g = *reinterpret_cast<const float4*>(a.w_norm + i);
        float sq = r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
#pragma unroll
        for (int cc = 0; cc < HC; ++cc) if (cc == c) ss[cc] += sq;
        *reinterpret_cast<float4*>(xn + i) = make_float4(r.x * g.x, r.y * g.y, r.z * g.z, r.w * g.w);
    }
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        const float v = warp_sum(ss[c]);
        if (lane == 0) part[warp][c] = v;
    }
    __syncthreads();
    if (t < HC) {
        float s = 0.0f;
        for (int w = 0; w < WARPS; ++w) s += part[w][t];
        s_rs[t] = rsqrtf(s / (float) N + a.eps);
        if (blockIdx.x == 0) a.rs[t] = s_rs[t];
    }
    __syncthreads();
    for (int i = t; i < D; i += THREADS) xn[i] *= s_rs[i / N];
    __syncthreads();
    // 2. one warp per output row: 10240 bf16 = 1280 chunks of 8, 40 per lane.
    const bool inject_block = blockIdx.x == DOWN_BLOCKS;
    const int row = inject_block ? warp : blockIdx.x * WARPS + warp;
    if (inject_block && (a.w_inject == nullptr || warp >= HC)) return;
    const uint16_t* wrow = (inject_block ? a.w_inject : a.w_down) + (size_t) row * D;
    const uint4* w4 = reinterpret_cast<const uint4*>(wrow);
    float acc = 0.0f;
#pragma unroll 4
    for (int j = lane; j < D / 8; j += 32) acc += dot8(__ldg(w4 + j), xn + j * 8);
    acc = warp_sum(acc);
    if (lane != 0) return;
    if (inject_block) {
        a.inject_out[row] = acc;
    } else {
        const float x = acc / (float) HC;
        a.lo[row] = x / (1.0f + __expf(-x));
    }
}

__global__ void __launch_bounds__(THREADS) gr_up_kernel(FusedGrArgs a) {
    __shared__ __align__(16) float lo[LR];
    __shared__ float g[HC][UP_COLS];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int d0 = blockIdx.x * UP_COLS;
    for (int k = t; k < LR; k += THREADS) lo[k] = a.lo[k];
    __syncthreads();
    // 128 rows (4 streams x 32 columns), 16 per warp: 320 bf16 = 40 chunks of 8.
    for (int r = warp; r < HC * UP_COLS; r += WARPS) {
        const int c = r / UP_COLS, dd = r - c * UP_COLS, i = c * N + d0 + dd;
        const uint4* w4 = reinterpret_cast<const uint4*>(a.w_up + (size_t) i * LR);
        float acc = dot8(__ldg(w4 + lane), lo + lane * 8);
        if (lane < LR / 8 - 32) acc += dot8(__ldg(w4 + 32 + lane), lo + (32 + lane) * 8);
        acc = warp_sum(acc);
        if (lane == 0) {
            float rv = a.R[i];
            if (a.apply) {
                rv = fmaf(a.bo_prev[d0 + dd], 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC), rv);
                a.R_out[i] = rv;                       // this block owns column d0+dd of every stream
            }
            const float x = rv * a.w_norm[i] * a.rs[c];
            g[c][dd] = x * sigmoidf_(acc);
        }
    }
    __syncthreads();
    if (t < UP_COLS) {
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) s += g[c][t];
        a.mixed[d0 + t] = s / (float) HC;
    }
}

// ================================ plan v0.3 P6: T tokens, one weight read ================================
struct GrMulti {
    FusedGrArgs a[kFusedGrMaxT];
    float* xn;
    int T;
};

// Step 1 of `gr_down_kernel`, one block per token, same threads and reduction order: rs[t] and xn[t] to global.
__global__ void __launch_bounds__(THREADS) gr_norm_multi_kernel(GrMulti m) {
    __shared__ float part[WARPS][HC];
    __shared__ float s_rs[HC];
    const FusedGrArgs& a = m.a[blockIdx.x];
    float* xn = m.xn + (size_t) blockIdx.x * D;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    float gw[HC];
#pragma unroll
    for (int c = 0; c < HC; ++c) gw[c] = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
    float ss[HC] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = t * 4; i < D; i += THREADS * 4) {
        const int c = i / N, d = i - c * N;
        float4 r = *reinterpret_cast<const float4*>(a.R + i);
        if (a.apply) {
            const float4 b = *reinterpret_cast<const float4*>(a.bo_prev + d);
            r.x = fmaf(b.x, gw[c], r.x); r.y = fmaf(b.y, gw[c], r.y);
            r.z = fmaf(b.z, gw[c], r.z); r.w = fmaf(b.w, gw[c], r.w);
        }
        const float4 g = *reinterpret_cast<const float4*>(a.w_norm + i);
        float sq = r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
#pragma unroll
        for (int cc = 0; cc < HC; ++cc) if (cc == c) ss[cc] += sq;
        *reinterpret_cast<float4*>(xn + i) = make_float4(r.x * g.x, r.y * g.y, r.z * g.z, r.w * g.w);
    }
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        const float v = warp_sum(ss[c]);
        if (lane == 0) part[warp][c] = v;
    }
    __syncthreads();
    if (t < HC) {
        float s = 0.0f;
        for (int w = 0; w < WARPS; ++w) s += part[w][t];
        s_rs[t] = rsqrtf(s / (float) N + a.eps);
        a.rs[t] = s_rs[t];
    }
    __syncthreads();
    for (int i = t; i < D; i += THREADS) xn[i] *= s_rs[i / N];
}

#if defined(__HIPCC__) && !defined(STRATA_HIP)   // gfx906 (STRATA_HIP) keeps 2560: its down path is the K-split kernel
constexpr int TILE = 1280;             // eight-token tile fits gfx1100's 64 KiB LDS limit
#else
constexpr int TILE = 2560;             // xn floats per token staged at a time: 320 chunks of 8, 10 per lane
#endif
constexpr int TQ = TILE / 8 / 32;      // uint4 weight chunks per lane per tile

// Step 2 of `gr_down_kernel` for T tokens.  One warp per row (so each lane accumulates the same chunks in the
// same order as the single-token kernel); per tile the lane's 10 weight chunks are loaded BEFORE the activation
// tile is staged, so the DRAM and L2 traffic are in flight together.
__global__ void __launch_bounds__(THREADS) gr_down_multi_kernel(GrMulti m) {
    extern __shared__ __align__(16) float tile[];   // [T][TILE]
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int T = m.T;
    const bool inject_block = blockIdx.x == DOWN_BLOCKS;
    const int row = inject_block ? warp : blockIdx.x * WARPS + warp;
    const bool active = !(inject_block && (m.a[0].w_inject == nullptr || warp >= HC));
    const uint16_t* wrow = (inject_block ? m.a[0].w_inject : m.a[0].w_down) + (size_t) (active ? row : 0) * D;
    const uint4* w4 = reinterpret_cast<const uint4*>(wrow);
    float acc[kFusedGrMaxT];
#pragma unroll
    for (int k = 0; k < kFusedGrMaxT; ++k) acc[k] = 0.0f;
    for (int base = 0; base < D; base += TILE) {
        uint4 wv[TQ];
        if (active) {
#pragma unroll
            for (int q = 0; q < TQ; ++q) wv[q] = __ldg(w4 + base / 8 + lane + 32 * q);
        }
        __syncthreads();                                   // the previous tile is consumed
        const float4* src4 = reinterpret_cast<const float4*>(m.xn);
        float4* tile4 = reinterpret_cast<float4*>(tile);
        for (int i = t; i < T * (TILE / 4); i += THREADS) {
            const int k = i / (TILE / 4), off = i - k * (TILE / 4);
            tile4[i] = src4[((size_t) k * D + base) / 4 + off];
        }
        __syncthreads();
        if (!active) continue;
#pragma unroll
        for (int q = 0; q < TQ; ++q) {
            const int j = lane + 32 * q;
#pragma unroll
            for (int k = 0; k < kFusedGrMaxT; ++k)
                if (k < T) acc[k] += dot8(wv[q], tile + k * TILE + j * 8);
        }
    }
    if (!active) return;
    float s[kFusedGrMaxT];
#pragma unroll
    for (int k = 0; k < kFusedGrMaxT; ++k) s[k] = k < T ? warp_sum(acc[k]) : 0.0f;
    // lane k writes token k (every lane holds every sum after the xor reduction)
#pragma unroll
    for (int k = 0; k < kFusedGrMaxT; ++k) {
        if (k >= T || lane != k) continue;
        if (inject_block) {
            m.a[k].inject_out[row] = s[k];
        } else {
            const float x = s[k] / (float) HC;
            m.a[k].lo[row] = x / (1.0f + __expf(-x));
        }
    }
}

constexpr int UPM_COLS = 16;                      // columns per block (x 4 streams = 64 rows, 8 per warp)
constexpr int UPM_BLOCKS = N / UPM_COLS;          // 160

// `gr_up_kernel` for T tokens: each row of w_up read once; the T dots reduced by xor so every lane holds every
// sum, and lane k runs token k's epilogue - the T epilogues in parallel instead of one after another.
__global__ void __launch_bounds__(THREADS) gr_up_multi_kernel(GrMulti m) {
    __shared__ __align__(16) float lo[kFusedGrMaxT][LR];
    __shared__ float g[kFusedGrMaxT][HC][UPM_COLS];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int T = m.T;
    const int d0 = blockIdx.x * UPM_COLS;
    for (int i = t; i < T * LR; i += THREADS) lo[i / LR][i % LR] = m.a[i / LR].lo[i % LR];
    __syncthreads();
    for (int r = warp; r < HC * UPM_COLS; r += WARPS) {
        const int c = r / UPM_COLS, dd = r - c * UPM_COLS, i = c * N + d0 + dd;
        const uint4* w4 = reinterpret_cast<const uint4*>(m.a[0].w_up + (size_t) i * LR);
        const uint4 wa = __ldg(w4 + lane);
        const uint4 wb = lane < LR / 8 - 32 ? __ldg(w4 + 32 + lane) : make_uint4(0, 0, 0, 0);
        // the epilogue inputs of this lane's token, fetched while the dots run
        float rv = 0.0f, wn = 0.0f, rsc = 0.0f, bo = 0.0f, ip = 0.0f;
        bool apply = false;
        if (lane < T) {
            const FusedGrArgs& a = m.a[lane];
            rv = a.R[i];
            wn = a.w_norm[i];
            rsc = a.rs[c];
            apply = a.apply;
            if (apply) { bo = a.bo_prev[d0 + dd]; ip = a.inj_prev[c]; }
        }
        float mine = 0.0f;
#pragma unroll
        for (int k = 0; k < kFusedGrMaxT; ++k) {
            if (k >= T) break;
            float acc = dot8(wa, lo[k] + lane * 8);
            if (lane < LR / 8 - 32) acc += dot8(wb, lo[k] + (32 + lane) * 8);
            acc = warp_sum(acc);
            if (lane == k) mine = acc;
        }
        if (lane < T) {
            if (apply) {
                rv = fmaf(bo, 2.0f * sigmoidf_(ip / (float) HC), rv);
                m.a[lane].R_out[i] = rv;
            }
            const float x = rv * wn * rsc;
            g[lane][c][dd] = x * sigmoidf_(mine);
        }
    }
    __syncthreads();
    for (int i = t; i < T * UPM_COLS; i += THREADS) {
        const int k = i / UPM_COLS, col = i - k * UPM_COLS;
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) s += g[k][c][col];
        m.a[k].mixed[d0 + col] = s / (float) HC;
    }
}


#if defined(STRATA_HIP)
// ---- AMD fast path (STRATA_GR_FAST, default 1; 0 = the old kernels): the same arithmetic per value in the same order, latency
// hidden.  gr_norm: one block per token (as before) but a thread's 10 float4 of R / bo / w_norm are loaded
// before any is used, and xn stays in registers until rs is known (was: store, then re-read the 40 KB).
// gr_up: a warp's 8 rows of w_up and their epilogue inputs are loaded before the dots (was: one row's load chain
// after another).
constexpr int NQ = D / (THREADS * 4);   // 10 float4 per thread
static_assert(D % (THREADS * 4) == 0, "whole float4 per thread");
__global__ void __launch_bounds__(THREADS) gr_norm_fast_kernel(GrMulti m) {
    __shared__ float part[WARPS][HC];
    __shared__ float s_rs[HC];
    const FusedGrArgs& a = m.a[blockIdx.x];
    float* xn = m.xn + (size_t) blockIdx.x * D;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    float gw[HC];
#pragma unroll
    for (int c = 0; c < HC; ++c) gw[c] = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
    float4 r[NQ], g[NQ];
#pragma unroll
    for (int k = 0; k < NQ; ++k) {
        const int i = t * 4 + k * THREADS * 4;
        r[k] = *reinterpret_cast<const float4*>(a.R + i);
        g[k] = *reinterpret_cast<const float4*>(a.w_norm + i);
    }
    if (a.apply) {
#pragma unroll
        for (int k = 0; k < NQ; ++k) {
            const int i = t * 4 + k * THREADS * 4, c = i / N, d = i - c * N;
            const float4 b = *reinterpret_cast<const float4*>(a.bo_prev + d);
            r[k].x = fmaf(b.x, gw[c], r[k].x); r[k].y = fmaf(b.y, gw[c], r[k].y);
            r[k].z = fmaf(b.z, gw[c], r[k].z); r[k].w = fmaf(b.w, gw[c], r[k].w);
        }
    }
    float ss[HC] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int k = 0; k < NQ; ++k) {
        const int i = t * 4 + k * THREADS * 4, c = i / N;
        const float sq = r[k].x * r[k].x + r[k].y * r[k].y + r[k].z * r[k].z + r[k].w * r[k].w;
#pragma unroll
        for (int cc = 0; cc < HC; ++cc) if (cc == c) ss[cc] += sq;
        r[k] = make_float4(r[k].x * g[k].x, r[k].y * g[k].y, r[k].z * g[k].z, r[k].w * g[k].w);
    }
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        const float v = warp_sum(ss[c]);
        if (lane == 0) part[warp][c] = v;
    }
    __syncthreads();
    if (t < HC) {
        float s = 0.0f;
        for (int w = 0; w < WARPS; ++w) s += part[w][t];
        s_rs[t] = rsqrtf(s / (float) N + a.eps);
        a.rs[t] = s_rs[t];
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < NQ; ++k) {
        const int i = t * 4 + k * THREADS * 4;
        const float sr = s_rs[i / N];
        *reinterpret_cast<float4*>(xn + i) = make_float4(r[k].x * sr, r[k].y * sr, r[k].z * sr, r[k].w * sr);
    }
}
// gr_up: 8 lanes per row instead of a 32-lane warp.  Lane j holds the old lanes j, j+8, j+16, j+24 (and chunk
// 32+j, the old lane j's second chunk): the xor tree's stages 16 and 8 are its own adds (a + b == b + a), stages
// 4, 2, 1 are shuffles inside the 8 - the same tree, bitwise the same sum, 3 shuffles a token instead of 5 and no
// idle lanes on the tail chunks.  A block's 64 rows are 2 passes of 32 groups, their weights loaded up front.
__device__ __forceinline__ float xor8(float v) {
#pragma unroll
    for (int o = 4; o > 0; o >>= 1) v += __shfl_xor(v, o, 64);
    return v;
}
__global__ void __launch_bounds__(THREADS) gr_up_fast_kernel(GrMulti m) {
    __shared__ __align__(16) float lo[kFusedGrMaxT][LR];
    __shared__ float g[kFusedGrMaxT][HC][UPM_COLS];
    const int t = threadIdx.x, j = t & 7, grp = t >> 3;
    const int T = m.T;
    const int d0 = blockIdx.x * UPM_COLS;
    static_assert(LR == 40 * 8 && HC * UPM_COLS == 64 && THREADS == 256, "geometry");
    uint4 w[2][5];
    float rv[2] = {0.0f, 0.0f}, wn[2] = {0.0f, 0.0f}, rsc[2] = {0.0f, 0.0f}, bo[2] = {0.0f, 0.0f}, ip[2] = {0.0f, 0.0f};
    const bool apply = j < T && m.a[j < T ? j : 0].apply;
#pragma unroll
    for (int p = 0; p < 2; ++p) {
        const int r = grp + 32 * p, c = r / UPM_COLS, dd = r - c * UPM_COLS, i = c * N + d0 + dd;
        const uint4* w4 = reinterpret_cast<const uint4*>(m.a[0].w_up + (size_t) i * LR);
#pragma unroll
        for (int q = 0; q < 4; ++q) w[p][q] = __ldg(w4 + j + 8 * q);
        w[p][4] = __ldg(w4 + 32 + j);
        if (j < T) {
            const FusedGrArgs& a = m.a[j];
            rv[p] = a.R[i];
            wn[p] = a.w_norm[i];
            rsc[p] = a.rs[c];
            if (apply) { bo[p] = a.bo_prev[d0 + dd]; ip[p] = a.inj_prev[c]; }
        }
    }
    for (int i = t; i < T * LR; i += THREADS) lo[i / LR][i % LR] = m.a[i / LR].lo[i % LR];
    __syncthreads();
#pragma unroll
    for (int p = 0; p < 2; ++p) {
        const int r = grp + 32 * p, c = r / UPM_COLS, dd = r - c * UPM_COLS, i = c * N + d0 + dd;
        float mine = 0.0f;
#pragma unroll
        for (int k = 0; k < kFusedGrMaxT; ++k) {
            if (k >= T) break;
            const float* l = lo[k];
            const float p0 = dot8(w[p][0], l + j * 8) + dot8(w[p][4], l + (32 + j) * 8);   // old lane j
            const float p1 = dot8(w[p][1], l + (j + 8) * 8);                               // old lane j + 8
            const float p2 = dot8(w[p][2], l + (j + 16) * 8);                              // old lane j + 16
            const float p3 = dot8(w[p][3], l + (j + 24) * 8);                              // old lane j + 24
            const float s = xor8((p0 + p2) + (p1 + p3));   // stage 16: (j, j+16), (j+8, j+24); stage 8; then 4, 2, 1
            if (j == k) mine = s;
        }
        if (j < T) {
            float x0 = rv[p];
            if (apply) {
                x0 = fmaf(bo[p], 2.0f * sigmoidf_(ip[p] / (float) HC), x0);
                m.a[j].R_out[i] = x0;
            }
            const float x = x0 * wn[p] * rsc[p];
            g[j][c][dd] = x * sigmoidf_(mine);
        }
    }
    __syncthreads();
    for (int i = t; i < T * UPM_COLS; i += THREADS) {
        const int k = i / UPM_COLS, col = i - k * UPM_COLS;
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) s += g[k][c][col];
        m.a[k].mixed[d0 + col] = s / (float) HC;
    }
}
#endif

#if defined(STRATA_HIP)
// ---- AMD: `gr_down_multi` split along K.  The CUDA kernel is one warp per row of w_down (320 + 4 rows), 41
// blocks: on a 60-CU gfx906 that left most of the card idle and read the 6.5 MB matrix at ~100 GB/s.  Here a block
// is 8 wavefronts = 8 rows of one K-slice of 2048 (the slice of every token's xn staged in LDS once), 41 x 5
// blocks, and a second kernel adds the 5 slice sums in a fixed order and runs the epilogue.  A token's result does
// not depend on how many tokens share the window (each column is reduced on its own).
constexpr int GS_SL = 2048;                   // K per slice
constexpr int GS_S = D / GS_SL;               // 5 slices
constexpr int GS_ROWS = LR + HC;              // 324: the down rows, then the inject rows
constexpr int GS_WAVES = 8;
static_assert(D % GS_SL == 0 && GS_SL == 64 * 8 * 4, "a lane takes 4 chunks of 8 per slice");
// the slice sums: a module-scope device array is per device by construction, and needs no allocation during a
// graph capture
__device__ float g_gr_part[GS_S * kFusedGrMaxT * GS_ROWS];
__global__ void __launch_bounds__(GS_WAVES * 64) gr_down_split_kernel(GrMulti m) {
    float* part = g_gr_part;
    extern __shared__ __align__(16) float tile[];   // [T][GS_SL]
    const int t = threadIdx.x, lane = t & 63, wave = t >> 6;
    const int T = m.T, sl = blockIdx.y;
    const bool inject_block = blockIdx.x == DOWN_BLOCKS;
    const int row = inject_block ? wave : blockIdx.x * GS_WAVES + wave;
    const bool active = !(inject_block && (m.a[0].w_inject == nullptr || wave >= HC));
    const uint16_t* wrow = (inject_block ? m.a[0].w_inject : m.a[0].w_down) + (size_t) (active ? row : 0) * D;
    const uint4* w4 = reinterpret_cast<const uint4*>(wrow) + sl * (GS_SL / 8);
    uint4 wv[4];
    if (active) {
#pragma unroll
        for (int q = 0; q < 4; ++q) wv[q] = __ldg(w4 + lane + 64 * q);
    }
    const float4* src4 = reinterpret_cast<const float4*>(m.xn);
    float4* tile4 = reinterpret_cast<float4*>(tile);
    for (int i = t; i < T * (GS_SL / 4); i += GS_WAVES * 64) {
        const int k = i / (GS_SL / 4), off = i - k * (GS_SL / 4);
        tile4[i] = src4[((size_t) k * D + (size_t) sl * GS_SL) / 4 + off];
    }
    __syncthreads();
    if (!active) return;
    float acc[kFusedGrMaxT];
#pragma unroll
    for (int k = 0; k < kFusedGrMaxT; ++k) acc[k] = 0.0f;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int j = lane + 64 * q;
#pragma unroll
        for (int k = 0; k < kFusedGrMaxT; ++k)
            if (k < T) acc[k] += dot8(wv[q], tile + k * GS_SL + j * 8);
    }
#pragma unroll
    for (int k = 0; k < kFusedGrMaxT; ++k) {
        if (k >= T) break;
        float v = acc[k];
#pragma unroll
        for (int off = 32; off > 0; off >>= 1) v += __shfl_xor(v, off, 64);
        if (lane == 0) part[((size_t) sl * kFusedGrMaxT + k) * GS_ROWS + (inject_block ? LR + row : row)] = v;
    }
}
__global__ void gr_down_finish_kernel(GrMulti m) {
    const float* part = g_gr_part;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = i / GS_ROWS, r = i - k * GS_ROWS;
    if (k >= m.T) return;
    const bool inject = r >= LR;
    if (inject && m.a[0].w_inject == nullptr) return;
    float s = 0.0f;
#pragma unroll
    for (int sl = 0; sl < GS_S; ++sl) s += part[((size_t) sl * kFusedGrMaxT + k) * GS_ROWS + r];
    if (inject) {
        m.a[k].inject_out[r - LR] = s;
    } else {
        const float x = s / (float) HC;
        m.a[k].lo[r] = x / (1.0f + __expf(-x));
    }
}

#endif
}  // namespace

#if defined(STRATA_HIP)
int g_gr_fast = -1;   // fused_gr_set_fast (the bench); -1 = STRATA_GR_FAST
static bool gr_fast() {
    static const bool env = [] { const char* v = std::getenv("STRATA_GR_FAST"); return v ? std::atoi(v) != 0 : true; }();
    return g_gr_fast >= 0 ? g_gr_fast != 0 : env;
}
#else
int g_gr_fast = -1;
#endif
void fused_gr_set_fast(int on) { g_gr_fast = on; }

void fused_gr_read_multi(const FusedGrArgs* a, int n_tok, float* xn_scratch, void* stream, unsigned long long* stamp_buf,
                         int stamp_i0) {
    if (n_tok < 1 || n_tok > kFusedGrMaxT || xn_scratch == nullptr) {
        std::fprintf(stderr, "fused_gr_read_multi: invalid arguments\n");
        std::exit(1);
    }
    GrMulti m;
    for (int t = 0; t < n_tok; ++t) {
        m.a[t] = a[t];
        const FusedGrArgs& x = a[t];
        if (!x.R || !x.w_norm || !x.w_down || !x.w_up || !x.lo || !x.rs || !x.mixed || (x.w_inject && !x.inject_out) ||
            (x.apply && (!x.bo_prev || !x.inj_prev || !x.R_out)) || x.w_down != a[0].w_down || x.w_up != a[0].w_up ||
            x.w_inject != a[0].w_inject || x.w_norm != a[0].w_norm) {
            std::fprintf(stderr, "fused_gr_read_multi: invalid arguments for token %d\n", t);
            std::exit(1);
        }
    }
    m.xn = xn_scratch;
    m.T = n_tok;
    cudaStream_t st = (cudaStream_t) stream;
#if defined(STRATA_HIP)
    const bool fast = gr_fast();
    if (fast) gr_norm_fast_kernel<<<n_tok, THREADS, 0, st>>>(m);
    else
#endif
    gr_norm_multi_kernel<<<n_tok, THREADS, 0, st>>>(m);
    if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0, stream);
#if defined(STRATA_HIP)
    static const bool split_off = std::getenv("STRATA_GR_SPLIT") && std::string(std::getenv("STRATA_GR_SPLIT")) == "0";
    if (!split_off) {
        gr_down_split_kernel<<<dim3(DOWN_BLOCKS + 1, GS_S), GS_WAVES * 64, (size_t) n_tok * GS_SL * sizeof(float), st>>>(
            m);
        gr_down_finish_kernel<<<(unsigned) ((n_tok * GS_ROWS + 255) / 256), 256, 0, st>>>(m);
        if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0 + 1, stream);
        if (fast) gr_up_fast_kernel<<<UPM_BLOCKS, THREADS, 0, st>>>(m);
        else gr_up_multi_kernel<<<UPM_BLOCKS, THREADS, 0, st>>>(m);
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            std::fprintf(stderr, "fused_gr_read_multi: %s\n", cudaGetErrorString(e));
            std::exit(1);
        }
        return;
    }
#endif
    // the shared-memory opt-in is a per-DEVICE setting: once per device, not once per process (a layer split
    // runs this kernel on two cards)
    static bool attr[64] = {};
    static int chunk[64] = {};   // Turing port: tokens the down kernel may carry in one launch on this card
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev >= 0 && dev < 64 && !attr[dev]) {
        // at most what the card allows (Turing: 64 KB - enough for windows of up to 6 tokens)
        int optin = 0;
        cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        int want = (int) (kFusedGrMaxT * TILE * sizeof(float));
        if (optin > 0 && want > optin) want = optin;
        cudaFuncSetAttribute(gr_down_multi_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, want);
        cudaGetLastError();      // drop any error the attempt left behind
        // Turing port: the down kernel stages n_tok*TILE floats of dynamic shared memory - 80 KB at the full
        // 8 tokens.  A card whose opt-in is below that (Turing: 64 KB, so 7+ tokens fail to launch as
        // "invalid argument") processes the tokens in slices that fit; a card that reports no opt-in gets what
        // fits the 48 KB default (4 tokens of the CUDA tile; all 8 of HIP's smaller tile).  The down kernel's
        // outputs (lo, inject_out) are strictly per-token, so the chunk boundaries are safe, and the up kernel
        // below still sees every token of the batch in one launch.
        const int capacity = (optin > 0 ? optin : 48 * 1024) / (int) (TILE * sizeof(float));
        chunk[dev] = capacity < 1 ? 1 : (capacity > kFusedGrMaxT ? kFusedGrMaxT : capacity);
        attr[dev] = true;
    }
    const int chunk_tok = (dev >= 0 && dev < 64 && chunk[dev]) ? chunk[dev] : kFusedGrMaxT;
    if (chunk_tok >= n_tok) {
        gr_down_multi_kernel<<<DOWN_BLOCKS + 1, THREADS, (size_t) n_tok * TILE * sizeof(float), st>>>(m);
    } else {
        for (int c0 = 0; c0 < n_tok; c0 += chunk_tok) {
            const int ct = n_tok - c0 < chunk_tok ? n_tok - c0 : chunk_tok;
            GrMulti c{};
            c.xn = xn_scratch + (size_t) c0 * D;
            c.T = ct;
            for (int k = 0; k < ct; ++k) c.a[k] = a[c0 + k];
            gr_down_multi_kernel<<<DOWN_BLOCKS + 1, THREADS, (size_t) ct * TILE * sizeof(float), st>>>(c);
        }
    }
    if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0 + 1, stream);
    gr_up_multi_kernel<<<UPM_BLOCKS, THREADS, 0, st>>>(m);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "fused_gr_read_multi: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

bool fused_gr_supported(int64_t n_embd, int64_t hc, int64_t hc_lr) {
    return n_embd == N && hc == HC && hc_lr == LR;
}

void fused_gr_read(const FusedGrArgs& a, void* stream) {
    if (!a.R || !a.w_norm || !a.w_down || !a.w_up || !a.lo || !a.rs || !a.mixed ||
        (a.w_inject && !a.inject_out) || (a.apply && (!a.bo_prev || !a.inj_prev || !a.R_out)) ||
        (a.apply && a.inj_prev == a.inject_out)) {
        std::fprintf(stderr, "fused_gr_read: invalid arguments\n");
        std::exit(1);
    }
    cudaStream_t st = (cudaStream_t) stream;
    gr_down_kernel<<<DOWN_BLOCKS + 1, THREADS, 0, st>>>(a);
    gr_up_kernel<<<UP_BLOCKS, THREADS, 0, st>>>(a);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "fused_gr_read: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace strata::kernels
