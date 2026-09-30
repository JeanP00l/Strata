// src/kernels/p2p_bench.cu - can the two cards sum a verify window's hidden state between them fast enough for a
// tensor split, and does it stay up?  A push-only all-reduce: each card WRITES its partial into the other card's
// fine-grained VRAM and raises a per-block flag there, then spins on ITS OWN memory for the other side's flag and
// adds.  Nothing ever reads across PCIe (llama.cpp's AllReduce hang on ROCm 6.3 looked like a wave waiting for a
// PCIe read that never came back).  A spin that waits too long gives up and counts a timeout instead of hanging.
//
//     build/p2p_bench [endurance_exchanges]      (both cards; 0 = skip the long run)
//
// Printed: hipMemcpyPeerAsync latency; the all-reduce alone, 96 in a graph (a window has 2 per layer x 48 layers);
// the same interleaved with ~1 MB of reads per step (compute next to the exchange), against the reads alone; then
// the endurance run with every element checked on both cards.
#include <cuda_runtime.h>
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>

#define CK(x)                                                                                    \
    do {                                                                                         \
        hipError_t e_ = (x);                                                                     \
        if (e_ != hipSuccess) {                                                                  \
            std::printf("%s:%d %s -> %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_));      \
            std::exit(1);                                                                        \
        }                                                                                        \
    } while (0)

constexpr int kN = 2560;          // hidden size
constexpr int kMaxT = 8;          // tokens in a window, upper bound
constexpr int kMaxBlocks = 64;
constexpr int kSteps = 96;        // exchanges per window

struct Side {
    int dev;
    float* x;            // local partial, kMaxT*kN
    float* out;          // local sum
    float* recv;         // 2 slots x kMaxT*kN, fine-grained, the other card writes here
    uint32_t* flags;     // 2 slots x kMaxBlocks, fine-grained, the other card writes here
    float* peer_recv;    // the other card's recv
    uint32_t* peer_flags;
    uint32_t* err;       // [0] timeouts, [1] wrong sums
    uint32_t* seq;       // exchange counter, advanced on the device so a graph can be replayed
    uint4* rd;           // read buffer for the compute stand-in
    float* sink;
    hipStream_t s;
};

// How a card reads a word the OTHER card wrote into its VRAM: P2P_LOAD=0 system-scope atomic, 1 glc slc asm
// (bypass L1 and L2 streaming), 2 volatile.
__device__ int g_load_mode;
__device__ uint32_t g_dbg[kMaxBlocks * 4];    // per block: spins, last flag seen, ticks waited, seq wanted
__device__ __forceinline__ uint32_t load_u32(const uint32_t* p) {
    if (g_load_mode == 1) {
        uint32_t v;
        asm volatile("global_load_dword %0, %1, off glc slc\n s_waitcnt vmcnt(0)" : "=v"(v) : "v"(p) : "memory");
        return v;
    }
    if (g_load_mode == 2) return *(const volatile uint32_t*) p;
    return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}

// How a card writes into the OTHER card's VRAM: P2P_STORE=0 plain/nontemporal + system release store,
// 1 glc slc asm, 2 system-scope atomic exchange for the flag (data as 1).
__device__ int g_store_mode;
__device__ __forceinline__ void store_u32(uint32_t* p, uint32_t v) {
    if (g_store_mode >= 1) {
        asm volatile("global_store_dword %0, %1, off glc slc" : : "v"(p), "v"(v) : "memory");
        return;
    }
    __builtin_nontemporal_store(v, p);
}
__device__ __forceinline__ void store_flag(uint32_t* p, uint32_t v) {
    if (g_store_mode == 2) {
        __hip_atomic_exchange(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        return;
    }
    if (g_store_mode == 1) {
        asm volatile("s_waitcnt vmcnt(0)\n global_store_dword %0, %1, off glc slc\n s_waitcnt vmcnt(0)" : : "v"(p), "v"(v)
                     : "memory");
        return;
    }
    __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ float fill_value(uint32_t seq, int dev, int i) {
    return (float) ((seq * 131u + (uint32_t) i * 7u + (uint32_t) dev * 1000003u) & 0xffffu) * 0.25f;
}

// Each step: produce the partial (deterministic from seq), push it, wait, sum, optionally check.
// seq_base lives in device memory: step k of launch j uses seq = seq_base + k + 1; the last block of the last step
// does not advance it - the host kernel `advance` does, once per graph launch.
__global__ void allreduce_push(float* x, float* out, float* peer_recv, uint32_t* peer_flags, const float* recv,
                               const uint32_t* flags, const uint32_t* seq_base, int step, int dev, int n, int check,
                               uint32_t* err) {
    const uint32_t seq = *seq_base + (uint32_t) step + 1u;
    const int slot = seq & 1u;
    const int per = (n + gridDim.x - 1) / gridDim.x;
    const int b0 = blockIdx.x * per;
    const int b1 = min(n, b0 + per);
    float* pr = peer_recv + (size_t) slot * kMaxT * kN;
    for (int i = b0 + threadIdx.x; i < b1; i += blockDim.x) {
        const float v = fill_value(seq, dev, i);
        x[i] = v;
        store_u32((uint32_t*) (pr + i), __float_as_uint(v));
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        store_flag(peer_flags + slot * kMaxBlocks + blockIdx.x, seq);
        const uint64_t t0 = wall_clock64();    // 25 MHz on gfx906
        uint32_t spins = 0, seen = 0;
        while ((seen = load_u32(flags + slot * kMaxBlocks + blockIdx.x)) < seq) {
            __builtin_amdgcn_s_sleep(1);
            ++spins;
            if (wall_clock64() - t0 > 5000000u) {    // 200 ms: give up, count it, do not hang the card
                atomicAdd(err, 1u);
                break;
            }
        }
        g_dbg[blockIdx.x * 4 + 0] = spins;
        g_dbg[blockIdx.x * 4 + 1] = seen;
        g_dbg[blockIdx.x * 4 + 2] = (uint32_t) (wall_clock64() - t0);
        g_dbg[blockIdx.x * 4 + 3] = seq;
    }
    __syncthreads();
    const float* r = recv + (size_t) slot * kMaxT * kN;
    for (int i = b0 + threadIdx.x; i < b1; i += blockDim.x) {
        const float mine = x[i];
        const float theirs = __uint_as_float(load_u32((const uint32_t*) (r + i)));
        const float sum = mine + theirs;
        out[i] = sum;
        if (check) {
            const float want = fill_value(seq, 0, i) + fill_value(seq, 1, i);
            if (sum != want) atomicAdd(err + 1, 1u);
        }
    }
}

__global__ void poke(float* peer_recv, uint32_t* peer_flags, int v) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < kN; i += gridDim.x * blockDim.x) peer_recv[i] = (float) v;
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0)
        __hip_atomic_store(peer_flags + blockIdx.x, (uint32_t) v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

__global__ void watch(const uint32_t* flags, uint32_t want) {
    const uint64_t t0 = wall_clock64();
    uint32_t spins = 0, seen = 0;
    while ((seen = load_u32(flags + blockIdx.x)) < want) {
        __builtin_amdgcn_s_sleep(1);
        ++spins;
        if (wall_clock64() - t0 > 5000000u) break;
    }
    if (threadIdx.x == 0) {
        g_dbg[blockIdx.x * 4 + 0] = spins;
        g_dbg[blockIdx.x * 4 + 1] = seen;
        g_dbg[blockIdx.x * 4 + 2] = (uint32_t) (wall_clock64() - t0);
    }
}

__global__ void advance(uint32_t* seq_base, uint32_t by) { *seq_base += by; }

__global__ void read_kernel(const uint4* __restrict__ p, size_t n, float* out) {
    uint32_t acc = 0;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        const uint4 v = p[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x12345678u) out[0] = 1.0f;
}

static void setup(Side& a, int dev) {
    a.dev = dev;
    CK(hipSetDevice(dev));
    CK(hipMalloc(&a.x, sizeof(float) * kMaxT * kN));
    CK(hipMalloc(&a.out, sizeof(float) * kMaxT * kN));
    // P2P_ALLOC=1: uncached (MTYPE UC on every mapping, the writer's L2 cannot hold the words); 0: fine-grained
    const char* al = std::getenv("P2P_ALLOC");
    const unsigned fl = (al && std::atoi(al) == 1) ? hipDeviceMallocUncached : hipDeviceMallocFinegrained;
    CK(hipExtMallocWithFlags((void**) &a.recv, sizeof(float) * 2 * kMaxT * kN, fl));
    CK(hipExtMallocWithFlags((void**) &a.flags, sizeof(uint32_t) * 2 * kMaxBlocks, fl));
    CK(hipMemset(a.flags, 0, sizeof(uint32_t) * 2 * kMaxBlocks));
    CK(hipMalloc(&a.err, 8));
    CK(hipMemset(a.err, 0, 8));
    CK(hipMalloc(&a.seq, 4));
    CK(hipMemset(a.seq, 0, 4));
    CK(hipMalloc(&a.rd, (size_t) 64 << 20));
    CK(hipMemset(a.rd, 1, (size_t) 64 << 20));
    CK(hipMalloc(&a.sink, 4));
    CK(hipStreamCreateWithFlags(&a.s, hipStreamNonBlocking));
}

// One graph per card: kSteps x ([reads of `rd_bytes`] + [all-reduce of T*kN]) + advance.
static hipGraphExec_t capture(Side& a, int T, int blocks, size_t rd_bytes, bool ar, int check) {
    CK(hipSetDevice(a.dev));
    hipGraph_t g;
    hipGraphExec_t ge;
    CK(hipStreamBeginCapture(a.s, hipStreamCaptureModeThreadLocal));
    const size_t rd_n = rd_bytes / 16;
    for (int k = 0; k < kSteps; ++k) {
        if (rd_n) read_kernel<<<120, 256, 0, a.s>>>(a.rd + (size_t) (k % 32) * ((size_t) 2 << 20) / 16, rd_n, a.sink);
        if (ar)
            allreduce_push<<<blocks, 256, 0, a.s>>>(a.x, a.out, a.peer_recv, a.peer_flags, a.recv, a.flags, a.seq, k,
                                                   a.dev, T * kN, check, a.err);
    }
    if (ar) advance<<<1, 1, 0, a.s>>>(a.seq, kSteps);
    CK(hipStreamEndCapture(a.s, &g));
    CK(hipGraphInstantiate(&ge, g, nullptr, nullptr, 0));
    return ge;
}

// Launch both graphs `reps` times; returns host wall time per launch in us.
static double run_pair(Side* sd, hipGraphExec_t g0, hipGraphExec_t g1, int reps) {
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) {
        CK(hipSetDevice(0));
        CK(hipGraphLaunch(g0, sd[0].s));
        CK(hipSetDevice(1));
        CK(hipGraphLaunch(g1, sd[1].s));
    }
    CK(hipSetDevice(0));
    CK(hipStreamSynchronize(sd[0].s));
    CK(hipSetDevice(1));
    CK(hipStreamSynchronize(sd[1].s));
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
}

static void read_err(Side* sd, uint32_t out[2][2]) {
    for (int d = 0; d < 2; ++d) {
        CK(hipSetDevice(d));
        CK(hipMemcpy(out[d], sd[d].err, 8, hipMemcpyDeviceToHost));
    }
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const long long endurance = argc > 1 ? std::atoll(argv[1]) : 0;
    int n = 0;
    CK(hipGetDeviceCount(&n));
    if (n < 2) {
        std::printf("need 2 cards, have %d\n", n);
        return 1;
    }
    int can01 = 0, can10 = 0;
    CK(hipDeviceCanAccessPeer(&can01, 0, 1));
    CK(hipDeviceCanAccessPeer(&can10, 1, 0));
    std::printf("peer access 0->1 %d, 1->0 %d\n", can01, can10);
    if (!can01 || !can10) return 1;
    const bool late = std::getenv("P2P_LATE") != nullptr;    // enable peer access after the allocations
    auto enable = [] {
        CK(hipSetDevice(0));
        CK(hipDeviceEnablePeerAccess(1, 0));
        CK(hipSetDevice(1));
        CK(hipDeviceEnablePeerAccess(0, 0));
    };
    if (!late) enable();
    Side sd[2];
    setup(sd[0], 0);
    setup(sd[1], 1);
    if (late) enable();
    std::printf("peer access enabled %s the allocations\n", late ? "after" : "before");
    {
        const char* m = std::getenv("P2P_LOAD");
        const int mode = m ? std::atoi(m) : 0;
        for (int d = 0; d < 2; ++d) {
            CK(hipSetDevice(d));
            CK(hipMemcpyToSymbol(HIP_SYMBOL(g_load_mode), &mode, sizeof(int)));
        }
        const char* w = std::getenv("P2P_STORE");
        const int smode = w ? std::atoi(w) : 0;
        for (int d = 0; d < 2; ++d) {
            CK(hipSetDevice(d));
            CK(hipMemcpyToSymbol(HIP_SYMBOL(g_store_mode), &smode, sizeof(int)));
        }
        std::printf("load mode %d, store mode %d\n", mode, smode);
    }
    sd[0].peer_recv = sd[1].recv;
    sd[0].peer_flags = sd[1].flags;
    sd[1].peer_recv = sd[0].recv;
    sd[1].peer_flags = sd[0].flags;

    // 00. each direction alone: card d writes a pattern into the other card's recv (plain and fine-grained flags),
    // host reads it back - no spinning anywhere
    for (int d = 0; d < 2; ++d) {
        const int o = 1 - d;
        CK(hipSetDevice(o));
        CK(hipMemset(sd[o].recv, 0, sizeof(float) * kN));
        CK(hipMemset(sd[o].flags, 0, sizeof(uint32_t) * 2 * kMaxBlocks));
        CK(hipDeviceSynchronize());
        CK(hipSetDevice(d));
        poke<<<4, 256, 0, sd[d].s>>>(sd[d].peer_recv, sd[d].peer_flags, 1000 + d);
        CK(hipStreamSynchronize(sd[d].s));
        CK(hipSetDevice(o));
        float r[4];
        uint32_t f[4];
        CK(hipMemcpy(r, sd[o].recv + 100, sizeof(r), hipMemcpyDeviceToHost));
        CK(hipMemcpy(f, sd[o].flags, sizeof(f), hipMemcpyDeviceToHost));
        std::printf("write %d->%d: recv[100..103] %g %g %g %g, flags %u %u %u %u\n", d, o, r[0], r[1], r[2], r[3], f[0],
                    f[1], f[2], f[3]);
        CK(hipMemset(sd[o].flags, 0, sizeof(uint32_t) * 2 * kMaxBlocks));
        CK(hipDeviceSynchronize());
    }

    // 000. card 0 spins on its own flag; card 1 writes it 20 ms later from a separate kernel
    for (int d = 0; d < 2; ++d) {
        const int o = 1 - d;
        CK(hipSetDevice(d));
        CK(hipMemset(sd[d].flags, 0, sizeof(uint32_t) * 2 * kMaxBlocks));
        CK(hipDeviceSynchronize());
        watch<<<4, 64, 0, sd[d].s>>>(sd[d].flags, 7);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        CK(hipSetDevice(o));
        const auto tp = std::chrono::steady_clock::now();
        poke<<<4, 256, 0, sd[o].s>>>(sd[o].peer_recv, sd[o].peer_flags, 7);
        CK(hipStreamSynchronize(sd[o].s));
        const double poke_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tp).count();
        uint32_t mid[4];
        CK(hipSetDevice(d));
        CK(hipMemcpyAsync(mid, sd[d].flags, sizeof(mid), hipMemcpyDeviceToHost, sd[0].s == sd[d].s ? sd[1 - d].s : sd[0].s));
        CK(hipStreamSynchronize(sd[d].s));
        std::printf("poke took %.1f ms on the host\n", poke_ms);
        uint32_t g[16];
        CK(hipMemcpyFromSymbol(g, HIP_SYMBOL(g_dbg), sizeof(g)));
        std::printf("watch on %d, poke from %d: spins %u seen %u ms %.1f\n", d, o, g[0], g[1], g[2] / 25000.0);
        CK(hipMemset(sd[d].flags, 0, sizeof(uint32_t) * 2 * kMaxBlocks));
        CK(hipDeviceSynchronize());
    }

    // 0. one exchange, plain launches (no graph), flags read back
    {
        for (int d = 0; d < 2; ++d) {
            CK(hipSetDevice(d));
            allreduce_push<<<4, 256, 0, sd[d].s>>>(sd[d].x, sd[d].out, sd[d].peer_recv, sd[d].peer_flags, sd[d].recv,
                                                   sd[d].flags, sd[d].seq, 0, d, kN, 1, sd[d].err);
        }
        for (int d = 0; d < 2; ++d) {
            CK(hipSetDevice(d));
            CK(hipStreamSynchronize(sd[d].s));
        }
        for (int d = 0; d < 2; ++d) {
            CK(hipSetDevice(d));
            uint32_t g[16];
            CK(hipMemcpyFromSymbol(g, HIP_SYMBOL(g_dbg), sizeof(g)));
            std::printf("card %d block0 spins %u seen %u ticks %u want %u | block1 spins %u seen %u ticks %u\n", d, g[0],
                        g[1], g[2], g[3], g[4], g[5], g[6]);
            uint32_t f[2 * kMaxBlocks];
            CK(hipMemcpy(f, sd[d].flags, sizeof(f), hipMemcpyDeviceToHost));
            std::printf("card %d flags slot1: %u %u %u %u\n", d, f[kMaxBlocks], f[kMaxBlocks + 1], f[kMaxBlocks + 2],
                        f[kMaxBlocks + 3]);
        }
        uint32_t e[2][2];
        read_err(sd, e);
        std::printf("single exchange: timeouts %u/%u, wrong %u/%u\n", e[0][0], e[1][0], e[0][1], e[1][1]);
        for (int d = 0; d < 2; ++d) {
            CK(hipSetDevice(d));
            CK(hipMemset(sd[d].err, 0, 8));
            advance<<<1, 1, 0, sd[d].s>>>(sd[d].seq, 1);
            CK(hipStreamSynchronize(sd[d].s));
        }
    }

    // 1. plain peer copy latency, stream-ordered
    {
        CK(hipSetDevice(0));
        hipEvent_t e0, e1;
        CK(hipEventCreate(&e0));
        CK(hipEventCreate(&e1));
        for (int T : {1, 3, 5}) {
            const size_t b = sizeof(float) * T * kN;
            for (int i = 0; i < 20; ++i) CK(hipMemcpyPeerAsync(sd[1].recv, 1, sd[0].x, 0, b, sd[0].s));
            CK(hipEventRecord(e0, sd[0].s));
            for (int i = 0; i < 200; ++i) CK(hipMemcpyPeerAsync(sd[1].recv, 1, sd[0].x, 0, b, sd[0].s));
            CK(hipEventRecord(e1, sd[0].s));
            CK(hipEventSynchronize(e1));
            float ms = 0;
            CK(hipEventElapsedTime(&ms, e0, e1));
            std::printf("hipMemcpyPeerAsync T=%d (%zu KB): %.1f us each\n", T, b >> 10, 1e3 * ms / 200);
        }
    }

    // 2. all-reduce alone and next to reads, in graphs on both cards at once
    for (int T : {1, 3, 5}) {
        for (int blocks : {8, 16, 32, 64}) {
            hipGraphExec_t a0 = capture(sd[0], T, blocks, 0, true, 1);
            hipGraphExec_t a1 = capture(sd[1], T, blocks, 0, true, 1);
            run_pair(sd, a0, a1, 5);
            const double us = run_pair(sd, a0, a1, 50);
            uint32_t e[2][2];
            read_err(sd, e);
            std::printf("AR only    T=%d blocks=%2d: %7.1f us/graph, %5.1f us/exchange  (timeouts %u/%u, wrong %u/%u)\n",
                        T, blocks, us, us / kSteps, e[0][0], e[1][0], e[0][1], e[1][1]);
            CK(hipGraphExecDestroy(a0));
            CK(hipGraphExecDestroy(a1));
        }
    }
    for (size_t rd : {(size_t) 1 << 20, (size_t) 2 << 20}) {
        const int T = 3, blocks = 16;
        hipGraphExec_t w0 = capture(sd[0], T, blocks, rd, false, 0);
        hipGraphExec_t w1 = capture(sd[1], T, blocks, rd, false, 0);
        hipGraphExec_t a0 = capture(sd[0], T, blocks, rd, true, 0);
        hipGraphExec_t a1 = capture(sd[1], T, blocks, rd, true, 0);
        run_pair(sd, w0, w1, 5);
        const double w = run_pair(sd, w0, w1, 50);
        run_pair(sd, a0, a1, 5);
        const double a = run_pair(sd, a0, a1, 50);
        std::printf("reads %zu MB x96: alone %7.1f us, with AR %7.1f us -> exchange costs %5.1f us each\n", rd >> 20,
                    w, a, (a - w) / kSteps);
    }

    // 3. endurance: every exchange checked on both cards
    if (endurance > 0) {
        const int T = 3, blocks = 16;
        hipGraphExec_t a0 = capture(sd[0], T, blocks, (size_t) 1 << 20, true, 1);
        hipGraphExec_t a1 = capture(sd[1], T, blocks, (size_t) 1 << 20, true, 1);
        const long long launches = (endurance + kSteps - 1) / kSteps;
        auto t0 = std::chrono::steady_clock::now();
        for (long long i = 0; i < launches; i += 100) {
            run_pair(sd, a0, a1, 100);
            if ((i / 100) % 50 == 0) {
                uint32_t e[2][2];
                read_err(sd, e);
                const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                std::printf("endurance %lld exchanges, %.0f s: timeouts %u/%u, wrong %u/%u\n", (i + 100) * kSteps, s,
                            e[0][0], e[1][0], e[0][1], e[1][1]);
            }
        }
        uint32_t e[2][2];
        read_err(sd, e);
        std::printf("endurance done %lld exchanges: timeouts %u/%u, wrong %u/%u\n", launches * kSteps, e[0][0],
                    e[1][0], e[0][1], e[1][1]);
    }
    return 0;
}
