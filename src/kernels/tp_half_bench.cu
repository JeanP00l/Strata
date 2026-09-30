// src/kernels/tp_half_bench.cu - what a tensor split would pay per card: every dense projection of a verify
// window with its real type and shape, whole against the half one card would hold (half the output rows, or half
// the input for the projections that sum heads back into n_embd), for T = 1..4 tokens.  Weights rotate through
// 8 copies so a projection is read from HBM, not from the 4 MB L2, as in a window that walks 48 layers.
//
//     build/tp_half_bench
//
// The numbers are per launch in a graph of 200 launches.  Contents are random: the time does not depend on them.
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace K = strata::kernels;

constexpr int kCopies = 8;
constexpr int kReps = 200;

static cudaStream_t s;
static cudaEvent_t e0, e1;

template <class F>
static double time_graph(F launch) {
    cudaGraph_t g;
    cudaGraphExec_t ge;
    cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal);
    for (int i = 0; i < kReps; ++i) launch(i % kCopies);
    cudaStreamEndCapture(s, &g);
    cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0);
    cudaGraphLaunch(ge, s);
    cudaStreamSynchronize(s);
    cudaEventRecord(e0, s);
    for (int r = 0; r < 5; ++r) cudaGraphLaunch(ge, s);
    cudaEventRecord(e1, s);
    cudaEventSynchronize(e1);
    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaGraphExecDestroy(ge);
    cudaGraphDestroy(g);
    return 1e3 * ms / (5 * kReps);
}

static void* rand_buf(size_t bytes) {
    std::vector<uint8_t> h(bytes);
    std::mt19937 rng(1);
    for (auto& b : h) b = (uint8_t) (rng() & 0x3f);    // small bytes: finite scales for every type
    void* d;
    cudaMalloc(&d, bytes);
    cudaMemcpy(d, h.data(), bytes, cudaMemcpyHostToDevice);
    return d;
}

struct Proj {
    const char* name;
    int type;
    int n_in, n_out;
    bool split_in;    // true: the half cuts n_in (heads summed back into n_embd), else n_out
    int per_window;   // launches per window: GDN 36 layers, QSA 12
};

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    cudaStreamCreate(&s);
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    // ggml ids: Q4_K=12, Q6_K=14, IQ4_XS=23, IQ4_NL=20
    const Proj P[] = {
        {"gdn qkv   Q6_K  2560->10240", 14, 2560, 10240, false, 36},
        {"gdn z     Q4_K  2560->6144 ", 12, 2560, 6144, false, 36},
        {"gdn out   Q6_K  6144->2560 ", 14, 6144, 2560, true, 36},
        {"qsa q     IQ4XS 2560->12288", 23, 2560, 12288, false, 12},
        {"qsa k     Q6_K  2560->512  ", 14, 2560, 512, false, 12},
        {"qsa out   Q4_K  6144->2560 ", 12, 6144, 2560, true, 12},
        {"shexp g/u Q4_K  2560->640  ", 12, 2560, 640, false, 96},
        {"shexp dn  IQ4NL 640->2560  ", 20, 640, 2560, true, 48},
    };
    float* y;
    cudaMalloc((void**) &y, (size_t) 8 * 16384 * 4);
    void* xq;
    cudaMalloc(&xq, K::native_q8_1_bytes(8192, 8));
    cudaMemset(xq, 0, K::native_q8_1_bytes(8192, 8));
    std::printf("%-30s %3s %9s %9s %6s %11s\n", "projection", "T", "whole us", "half us", "ratio", "win ms w/h");
    double tot[5][2] = {};
    for (const Proj& p : P) {
        const size_t wb = K::native_mmvq_weight_bytes(p.type, p.n_in, p.n_out);
        void* w[kCopies];
        for (auto& b : w) b = rand_buf(wb);
        for (int T : {1, 2, 3, 4}) {
            const double full = time_graph([&](int c) { K::native_mmvq(p.type, w[c], xq, y, p.n_in, p.n_out, T, s); });
            const int hi = p.split_in ? p.n_in / 2 : p.n_in, ho = p.split_in ? p.n_out : p.n_out / 2;
            // a half with n_in cut reads a contiguous half of each row only if rows were re-packed; the bench uses
            // a matrix of that shape, which is what a card would hold after re-packing at load
            const double half = time_graph([&](int c) { K::native_mmvq(p.type, w[c], xq, y, hi, ho, T, s); });
            std::printf("%-30s %3d %9.1f %9.1f %6.2f %5.2f/%5.2f\n", p.name, T, full, half, half / full,
                        full * p.per_window / 1e3, half * p.per_window / 1e3);
            tot[T][0] += full * p.per_window / 1e3;
            tot[T][1] += half * p.per_window / 1e3;
        }
        for (auto& b : w) cudaFree(b);
    }
    // hyper-connections: 4 reads-worth of bf16 per layer (down 10240->320 and up 320->10240, twice), 48 layers.
    // The split owns half of the n_embd columns of all 4 streams: down cuts n_in, up cuts n_out.
    {
        const size_t wb = (size_t) 10240 * 320 * 2;
        void* w[kCopies];
        for (auto& b : w) b = rand_buf(wb);
        float* x;
        cudaMalloc((void**) &x, (size_t) 8 * 10240 * 4);
        cudaMemset(x, 0, (size_t) 8 * 10240 * 4);
        for (int T : {1, 2, 3, 4}) {
            auto dn = [&](int n_in) {
                return time_graph([&](int c) {
                    K::bf16_gemv_fp32_mmvf_multi(x, n_in, (const uint16_t*) w[c], y, 320, n_in, 320, T, s);
                });
            };
            auto up = [&](int n_out) {
                return time_graph([&](int c) {
                    K::bf16_gemv_fp32_mmvf_multi(x, 320, (const uint16_t*) w[c], y, n_out, 320, n_out, T, s);
                });
            };
            const double df = dn(10240), dh = dn(5120), uf = up(10240), uh = up(5120);
            std::printf("%-30s %3d %9.1f %9.1f %6.2f %5.2f/%5.2f\n", "hc down BF16 10240->320", T, df, dh, dh / df,
                        df * 96 / 1e3, dh * 96 / 1e3);
            std::printf("%-30s %3d %9.1f %9.1f %6.2f %5.2f/%5.2f\n", "hc up   BF16 320->10240", T, uf, uh, uh / uf,
                        uf * 96 / 1e3, uh * 96 / 1e3);
            tot[T][0] += (df + uf) * 96 / 1e3;
            tot[T][1] += (dh + uh) * 96 / 1e3;
        }
        // the engine's fused read for reference (whole only): what it really pays today for the same bytes
        K::FusedGrArgs a[4];
        float *R, *ws;
        cudaMalloc((void**) &R, (size_t) 4 * 10240 * 4);
        cudaMalloc((void**) &ws, (size_t) 4 * 16384 * 4);
        cudaMemset(R, 0, (size_t) 4 * 10240 * 4);
        void* wn = rand_buf(10240 * 4);
        void* winj = rand_buf(10240 * 4 * 2);
        for (int T : {1, 3}) {
            const double t = time_graph([&](int c) {
                for (int i = 0; i < T; ++i) {
                    a[i] = {};
                    a[i].R = R + (size_t) i * 10240;
                    a[i].R_out = R + (size_t) i * 10240;
                    a[i].w_norm = (const float*) wn;
                    a[i].w_down = (const uint16_t*) w[c];
                    a[i].w_up = (const uint16_t*) w[(c + 1) % kCopies];
                    a[i].w_inject = (const uint16_t*) winj;
                    a[i].lo = ws + (size_t) i * 4096;
                    a[i].rs = ws + (size_t) i * 4096 + 512;
                    a[i].inject_out = ws + (size_t) i * 4096 + 600;
                    a[i].mixed = ws + (size_t) i * 4096 + 1024;
                }
                K::fused_gr_read_multi(a, T, x, s);
            });
            std::printf("fused_gr_read_multi (engine)    T=%d %9.1f us, x96 = %.2f ms/window\n", T, t, t * 96 / 1e3);
        }
    }
    for (int T : {1, 2, 3, 4})
        std::printf("TOTAL dense per window T=%d: whole %.2f ms, half %.2f ms (%.2f)\n", T, tot[T][0], tot[T][1],
                    tot[T][1] / tot[T][0]);
    return 0;
}
